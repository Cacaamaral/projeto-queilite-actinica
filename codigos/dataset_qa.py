import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class QADataset(Dataset):
    # --- BLOCO 1: INICIALIZACAO ---
    def __init__(self, csv_file, transform=None):
        """
        Inicializa o dataset multimodal para a Queilite Actinica.
        csv_file: Caminho absoluto para o dataset_qa_limpo.csv
        transform: Transformacoes de imagem do torchvision (Resize, Normalize)
        """
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform
        
        # Filtra o dataset para remover o laudo invalido ('X')
        self.metadata = self.metadata[self.metadata['LAUDO'].isin(['SIM', 'NÃO'])].reset_index(drop=True)

    # --- BLOCO 2: PROCESSAMENTO CLINICO (TEXTO PARA MATRIZ NUMERICA) ---
    def _processar_clinicos(self, row):
        """
        Converte as categorias textuais de uma linha do CSV para um vetor numerico (Tensor).
        O tamanho final deste vetor sera de 10 posicoes (10 features clinicas).
        """
        features = []
        
        # 1. IDADE (Normalizacao simples dividindo por 100 para manter entre 0 e 1)
        idade = float(row['IDADE']) if pd.notna(row['IDADE']) else 50.0
        features.append(idade / 100.0)
        
        # 2. SEXO (1 para Masculino, 0 para Feminino)
        sexo = 1.0 if str(row['SEXO']).strip() == 'MASCULINO' else 0.0
        features.append(sexo)
        
        # 3. COR (Vetorizacao de 3 posicoes para separar as categorias)
        cor_str = str(row['COR']).strip()
        cor_branco = 1.0 if cor_str == 'BRANCO' else 0.0
        cor_pardo = 1.0 if cor_str == 'PARDO' else 0.0
        cor_preto = 1.0 if cor_str == 'PRETO' else 0.0
        features.extend([cor_branco, cor_pardo, cor_preto])
        
        # 4. TABAGISTA (Vetorizacao de 2 posicoes: SIM e EX)
        # Se ambos forem 0, significa 'NÃO' ou 'NAO_INFORMADO'
        tab_str = str(row['TABAGISTA']).strip()
        tab_sim = 1.0 if tab_str == 'SIM' else 0.0
        tab_ex = 1.0 if tab_str == 'EX' else 0.0
        features.extend([tab_sim, tab_ex])
        
        # 5. ETILISTA (Vetorizacao de 2 posicoes: SIM e EX)
        eti_str = str(row['ETILISTA']).strip()
        eti_sim = 1.0 if eti_str == 'SIM' else 0.0
        eti_ex = 1.0 if eti_str == 'EX' else 0.0
        features.extend([eti_sim, eti_ex])
        
        # 6. EXPOSICAO SOLAR (1 para SIM, 0 para NÃO/NAO_INFORMADO)
        expo_str = str(row['EXPOSIÇÃO']).strip()
        expo_sim = 1.0 if expo_str == 'SIM' else 0.0
        features.append(expo_sim)
        
        # Retorna um tensor do PyTorch com as 10 caracteristicas criadas
        return torch.tensor(features, dtype=torch.float32)

    # --- BLOCO 3: METODOS OBRIGATORIOS DO PYTORCH ---
    def __len__(self):
        # Retorna o tamanho total do dataset (numero de pacientes validos)
        return len(self.metadata)

    def __getitem__(self, idx):
        # Pega a linha do paciente especifico
        row = self.metadata.iloc[idx]
        
        # Leitura da Imagem
        img_path = row['CAMINHO_IMAGEM']
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # Imagem de seguranca caso o arquivo esteja corrompido
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # Leitura dos Dados Clinicos
        clinical_features = self._processar_clinicos(row)
        
        # Definicao do Rotulo Alvo (Label)
        # 1 = Tem Queilite Actinica (SIM), 0 = Nao tem (NÃO)
        label_str = str(row['LAUDO']).strip()
        label = 1 if label_str == 'SIM' else 0
        
        return image, clinical_features, label