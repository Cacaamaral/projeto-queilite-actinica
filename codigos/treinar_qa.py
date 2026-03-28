import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import os
import time
import numpy as np

# Importa a classe de manipulacao de dados desenvolvida anteriormente
from dataset_qa import QADataset

# --- BLOCO 1: DEFINICAO DA ARQUITETURA HIBRIDA (QA) ---
class ModeloHibridoQA(nn.Module):
    def __init__(self, num_clinical_features=10, num_classes=2):
        super(ModeloHibridoQA, self).__init__()
        
        # Inicializa a base visual com pesos da ImageNet para Transfer Learning
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Congela os tensores da base convolucional para evitar destruicao de padroes
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Remove a camada de saida original para extrair o vetor de 2048 caracteristicas
        num_visual_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # Constroi o perceptron multicamadas para os dados tabulares (Idade, Sexo, etc)
        self.clinical_branch = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Late Fusion: Agrupa o vetor visual (2048) e o vetor clinico (16)
        total_features = num_visual_features + 16
        
        # Camada densa final de decisao para classificacao binaria
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical_data):
        # Processamento paralelo
        visual_features = self.resnet(image)
        clinical_features = self.clinical_branch(clinical_data)
        
        # Fusao geometrica (concatenacao ao longo do eixo das colunas)
        fused_features = torch.cat((visual_features, clinical_features), dim=1)
        output = self.classifier(fused_features)
        
        return output

# --- BLOCO 2: INICIALIZACAO E HARDWARE ---
def main():
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')
    
    # Alocacao dinamica de processamento (GPU prioritariamente)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pipeline de pre-processamento visual (Redimensionamento e Normalizacao Estandardizada)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- BLOCO 3: DIVISAO DO DATASET E ISOLAMENTO DE AMOSTRAS ---
    dataset_completo = QADataset(csv_file=CSV_FILE, transform=data_transform)
    total_pacientes = len(dataset_completo)
    
    # Proporcao de Pareto mitigada (80/20)
    tamanho_treino = int(0.8 * total_pacientes)
    tamanho_val = total_pacientes - tamanho_treino
    
    # Semente estatica (seed 42) para reprodutibilidade rigorosa da validacao
    dataset_treino, dataset_val = random_split(
        dataset_completo, 
        [tamanho_treino, tamanho_val],
        generator=torch.Generator().manual_seed(42)
    )

    # --- BLOCO 4: BALANCEAMENTO MATEMATICO (OVERSAMPLING INDUZIDO) ---
    # Varredura dos laudos exclusivamente no conjunto isolado de treinamento
    rotulos_treino = [dataset_completo.metadata.iloc[idx]['LAUDO'] for idx in dataset_treino.indices]
    
    # Vetorizacao dos diagnosticos (1 = SIM, 0 = NAO)
    rotulos_numericos = [1 if str(laudo).strip() == 'SIM' else 0 for laudo in rotulos_treino]
    
    # Contabiliza frequencias absolutas de cada classe
    contagem_classes = np.bincount(rotulos_numericos)
    
    # Aplica penalizacao inversa (peso maior para a classe deficitaria)
    pesos_classes = 1. / contagem_classes
    
    # Mapeia o peso individual para o indice de cada tensor
    pesos_amostras = np.array([pesos_classes[t] for t in rotulos_numericos])
    pesos_amostras = torch.from_numpy(pesos_amostras).float()
    
    # Define o amostrador com reposicao ativada (replacement=True)
    amostrador_treino = WeightedRandomSampler(
        weights=pesos_amostras,
        num_samples=len(pesos_amostras),
        replacement=True 
    )

    # --- BLOCO 5: DATA LOADERS ---
    dataloaders = {
        # 'shuffle=True' removido pois entra em conflito com o uso do amostrador
        'train': DataLoader(dataset_treino, batch_size=16, sampler=amostrador_treino),
        'val': DataLoader(dataset_val, batch_size=16, shuffle=False)
    }
    dataset_sizes = {'train': tamanho_treino, 'val': tamanho_val}

    # --- BLOCO 6: INSTANCIACAO DA REDE E OTIMIZADORES ---
    model = ModeloHibridoQA(num_clinical_features=10, num_classes=2)
    model = model.to(device)

    # A ponderacao agora e feita no DataLoader, logo a Loss permanece inalterada
    criterion = nn.CrossEntropyLoss()
    
    # Backpropagation direcionado apenas para a arquitetura densa (Ramo Clinico + Classificador)
    optimizer = optim.Adam([
        {'params': model.clinical_branch.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001)

    # --- BLOCO 7: LOOP ITERATIVO DE TREINAMENTO ---
    since = time.time()
    num_epochs = 15 

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

            # Extracao de lotes pareados multimodalmente
            for inputs_img, inputs_clin, labels in dataloaders[phase]:
                inputs_img = inputs_img.to(device)
                inputs_clin = inputs_clin.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Calculo de gradientes restrito a fase de treinamento
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
    print(f'\nTreinamento QA concluido em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # --- BLOCO 8: EXPORTACAO DOS PESOS DO MODELO ---
    caminho_salvar = os.path.join(DIRETORIO_CODIGOS, 'modelo_qa_hibrido.pth')
    torch.save(model.state_dict(), caminho_salvar)

if __name__ == '__main__':
    main()