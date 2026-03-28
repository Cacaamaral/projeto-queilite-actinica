import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import time
import numpy as np

from dataset_qa import QADataset

# --- BLOCO 1: ARQUITETURA HIBRIDA COM GARGALO VISUAL ---
class ModeloHibridoQA(nn.Module):
    def __init__(self, num_clinical_features=10, num_classes=2):
        super(ModeloHibridoQA, self).__init__()
        
        # --- Ramo Visual (ResNet-50) ---
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        num_visual_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # NOVO: Gargalo Visual (Comprime 2048 para 32)
        # Impede que a imagem esmague os dados clinicos na fusao
        self.visual_bottleneck = nn.Sequential(
            nn.Linear(num_visual_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # --- Ramo Clinico ---
        # Mantem a saida em 16 caracteristicas
        self.clinical_branch = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- Fusao e Classificacao ---
        # Agora a proporcao e justa: 32 (Visual) + 16 (Clinico) = 48 tensores
        total_features = 32 + 16
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, image, clinical_data):
        raw_visual = self.resnet(image)
        # Passa os dados brutos da ResNet pelo gargalo
        compressed_visual = self.visual_bottleneck(raw_visual)
        
        clinical_features = self.clinical_branch(clinical_data)
        
        fused_features = torch.cat((compressed_visual, clinical_features), dim=1)
        output = self.classifier(fused_features)
        
        return output

# --- BLOCO 2: INICIALIZACAO E HARDWARE ---
def main():
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- BLOCO 3: PREPARACAO DO DATASET ---
    dataset_completo = QADataset(csv_file=CSV_FILE, transform=data_transform)
    total_pacientes = len(dataset_completo)
    
    tamanho_treino = int(0.8 * total_pacientes)
    tamanho_val = total_pacientes - tamanho_treino
    
    dataset_treino, dataset_val = random_split(
        dataset_completo, 
        [tamanho_treino, tamanho_val],
        generator=torch.Generator().manual_seed(42)
    )

    # --- BLOCO 4: BALANCEAMENTO (SAMPLER) ---
    rotulos_treino = [dataset_completo.metadata.iloc[idx]['LAUDO'] for idx in dataset_treino.indices]
    rotulos_numericos = [1 if str(laudo).strip() == 'SIM' else 0 for laudo in rotulos_treino]
    
    contagem_classes = np.bincount(rotulos_numericos)
    pesos_classes = 1. / contagem_classes
    pesos_amostras = np.array([pesos_classes[t] for t in rotulos_numericos])
    pesos_amostras = torch.from_numpy(pesos_amostras).float()
    
    amostrador_treino = WeightedRandomSampler(
        weights=pesos_amostras,
        num_samples=len(pesos_amostras),
        replacement=True 
    )

    # --- BLOCO 5: DATA LOADERS ---
    dataloaders = {
        'train': DataLoader(dataset_treino, batch_size=16, sampler=amostrador_treino),
        'val': DataLoader(dataset_val, batch_size=16, shuffle=False)
    }
    dataset_sizes = {'train': tamanho_treino, 'val': tamanho_val}

    # --- BLOCO 6: INSTANCIACAO E OTIMIZADOR ---
    model = ModeloHibridoQA(num_clinical_features=10, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # IMPORTANTE: O otimizador agora treina o Gargalo Visual, o Ramo Clinico e o Classificador
    # weight_decay (L2 Regularization) adicionado para impedir que os pesos viciem em uma unica classe
    optimizer = optim.Adam([
        {'params': model.visual_bottleneck.parameters()},
        {'params': model.clinical_branch.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001, weight_decay=1e-4)

    # --- BLOCO 7: LOOP DE TREINAMENTO ---
    since = time.time()
    num_epochs = 20 # Aumentado para 20 epocas devido a maior complexidade do Gargalo

    for epoch in range(num_epochs):
        print(f'\nEpoca {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs_img, inputs_clin, labels in dataloaders[phase]:
                inputs_img = inputs_img.to(device)
                inputs_clin = inputs_clin.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_img, inputs_clin)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs_img.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTreinamento concluido em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # --- BLOCO 8: SALVAMENTO ---
    caminho_salvar = os.path.join(DIRETORIO_CODIGOS, 'modelo_qa_hibrido.pth')
    torch.save(model.state_dict(), caminho_salvar)

if __name__ == '__main__':
    main()