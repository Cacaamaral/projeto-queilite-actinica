import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time

# Importa a classe customizada que processa as imagens e o CSV da Queilite Actinica
from dataset_qa import QADataset

# ---------------------------------------------------------
# BLOCO 1: DEFINICAO DA ARQUITETURA HIBRIDA (QA)
# ---------------------------------------------------------
class ModeloHibridoQA(nn.Module):
    def __init__(self, num_clinical_features=10, num_classes=2):
        super(ModeloHibridoQA, self).__init__()
        
        # --- Ramo Visual (ResNet-50) ---
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Congelamento dos pesos (Transfer Learning base)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Extracao bruta das 2048 caracteristicas visuais
        num_visual_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # --- Ramo Clinico (Dados Tabulares) ---
        # Recebe os 10 tensores do CSV (Idade, Sexo, Cor, Tabagismo, Etilismo, Sol)
        self.clinical_branch = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # --- Ramo de Fusao (Late Fusion) ---
        total_features = num_visual_features + 16
        
        # Camada decisoria final (SIM ou NAO para Queilite Actinica)
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical_data):
        visual_features = self.resnet(image)
        clinical_features = self.clinical_branch(clinical_data)
        
        # Concatenacao dos tensores
        fused_features = torch.cat((visual_features, clinical_features), dim=1)
        output = self.classifier(fused_features)
        
        return output

# ---------------------------------------------------------
# BLOCO 2: FUNCAO PRINCIPAL E CONFIGURACOES
# ---------------------------------------------------------
def main():
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')
    
    print("Iniciando treinamento do Modelo Hibrido para Queilite Actinica...")

    if not os.path.exists(CSV_FILE):
        print("ERRO: O arquivo dataset_qa_limpo.csv nao foi encontrado.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processamento alocado em: {device}")

    # ---------------------------------------------------------
    # BLOCO 3: PREPARACAO DOS DADOS E SPLIT (TREINO/VALIDACAO)
    # ---------------------------------------------------------
    # Padronizacao das imagens para a ResNet
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Ligeira rotacao para data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Instancia o dataset completo
    dataset_completo = QADataset(csv_file=CSV_FILE, transform=data_transform)
    total_pacientes = len(dataset_completo)
    
    # Divisao do dataset: 80% para treino, 20% para validacao
    tamanho_treino = int(0.8 * total_pacientes)
    tamanho_val = total_pacientes - tamanho_treino
    
    # Fixa a semente (seed) para que a divisao seja reprodutivel
    dataset_treino, dataset_val = random_split(
        dataset_completo, 
        [tamanho_treino, tamanho_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total de pacientes validos: {total_pacientes}")
    print(f"Pacientes no Treino: {tamanho_treino} | Validacao: {tamanho_val}")

    # Criacao dos DataLoaders
    # Batch size reduzido para 16 devido ao tamanho menor do dataset
    dataloaders = {
        'train': DataLoader(dataset_treino, batch_size=16, shuffle=True),
        'val': DataLoader(dataset_val, batch_size=16, shuffle=False)
    }
    dataset_sizes = {'train': tamanho_treino, 'val': tamanho_val}

    # ---------------------------------------------------------
    # BLOCO 4: INICIALIZACAO DO MODELO E OTIMIZADOR
    # ---------------------------------------------------------
    # 10 variaveis clinicas de entrada e 2 classes de saida (0 e 1)
    model = ModeloHibridoQA(num_clinical_features=10, num_classes=2)
    model = model.to(device)

    # Funcao de perda sem pesos forçados (conforme estrategia definida)
    criterion = nn.CrossEntropyLoss()
    
    # O otimizador ajusta apenas os pesos do ramo clinico e do classificador final
    optimizer = optim.Adam([
        {'params': model.clinical_branch.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=0.001)

    # ---------------------------------------------------------
    # BLOCO 5: LOOP DE TREINAMENTO
    # ---------------------------------------------------------
    since = time.time()
    num_epochs = 15 # Aumentado levemente devido ao menor volume de dados

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

            # Recebe a imagem, as 10 variaveis clinicas e o laudo (0 ou 1)
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
    print(f'\nTreinamento QA concluido em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # ---------------------------------------------------------
    # BLOCO 6: SALVAMENTO DO MODELO
    # ---------------------------------------------------------
    caminho_salvar = os.path.join(DIRETORIO_CODIGOS, 'modelo_qa_hibrido.pth')
    torch.save(model.state_dict(), caminho_salvar)
    print(f"Modelo final salvo em: {caminho_salvar}")

if __name__ == '__main__':
    main()