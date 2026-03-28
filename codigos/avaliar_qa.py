import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

from dataset_qa import QADataset

# --- BLOCO 1: ARQUITETURA HIBRIDA (ATUALIZADA COM GARGALO) ---
class ModeloHibridoQA(nn.Module):
    def __init__(self, num_clinical_features=10, num_classes=2):
        super(ModeloHibridoQA, self).__init__()
        
        # Ramo Visual (Sem pesos pré-treinados, pois carregaremos os nossos)
        self.resnet = models.resnet50(weights=None)
        num_visual_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # O Gargalo Visual exato que treinamos
        self.visual_bottleneck = nn.Sequential(
            nn.Linear(num_visual_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Ramo Clinico atualizado
        self.clinical_branch = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusao justa: 32 + 16 = 48
        total_features = 32 + 16
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes)
        )

    def forward(self, image, clinical_data):
        raw_visual = self.resnet(image)
        compressed_visual = self.visual_bottleneck(raw_visual)
        clinical_features = self.clinical_branch(clinical_data)
        fused_features = torch.cat((compressed_visual, clinical_features), dim=1)
        output = self.classifier(fused_features)
        return output

def main():
    # --- BLOCO 2: CONFIGURACOES E CARREGAMENTO DE DADOS ---
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')
    CAMINHO_MODELO = os.path.join(DIRETORIO_CODIGOS, 'modelo_qa_hibrido.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_completo = QADataset(csv_file=CSV_FILE, transform=data_transform_val)
    total_pacientes = len(dataset_completo)
    
    tamanho_treino = int(0.8 * total_pacientes)
    tamanho_val = total_pacientes - tamanho_treino
    
    _, dataset_val = random_split(
        dataset_completo, 
        [tamanho_treino, tamanho_val],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False)
    class_names = ['NÃO (Saudável)', 'SIM (Doente)']

    # --- BLOCO 3: CARREGAMENTO DOS PESOS ---
    model = ModeloHibridoQA(num_clinical_features=10, num_classes=2)

    if not os.path.exists(CAMINHO_MODELO):
        print("ERRO: O arquivo modelo_qa_hibrido.pth nao foi encontrado.")
        return

    # Agora a arquitetura estrutural e o arquivo de pesos correspondem perfeitamente
    state_dict = torch.load(CAMINHO_MODELO, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # --- BLOCO 4: INFERENCIA ---
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs_img, inputs_clin, labels in val_loader:
            inputs_img = inputs_img.to(device)
            inputs_clin = inputs_clin.to(device)
            labels = labels.to(device)

            outputs = model(inputs_img, inputs_clin)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --- BLOCO 5: GERACAO DE METRICAS ---
    print("\n--- RELATORIO DE CLASSIFICACAO FINAL (QA) ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predito (Inteligencia Artificial)')
    plt.ylabel('Real (Laudo Medico)')
    plt.title('Matriz de Confusao - QA (Gargalo Visual)')
    
    caminho_matriz = os.path.join(DIRETORIO_CODIGOS, 'matriz_confusao_qa.png')
    plt.savefig(caminho_matriz, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()