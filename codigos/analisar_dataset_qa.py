import pandas as pd
import os

def main():
    # --- BLOCO 1: CARREGAMENTO DO DATASET LIMPO ---
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    ARQUIVO_LIMPO = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')

    if not os.path.exists(ARQUIVO_LIMPO):
        print("ERRO: O arquivo dataset_qa_limpo.csv nao foi encontrado.")
        return

    df = pd.read_csv(ARQUIVO_LIMPO)
    print(f"Dataset carregado com sucesso. Total de pacientes prontos: {len(df)}\n")

    # --- BLOCO 2: ANALISE DA VARIAVEL ALVO (LAUDO) ---
    print("-" * 40)
    print("DISTRIBUICAO DOS DIAGNOSTICOS (LAUDO):")
    print("-" * 40)
    
    if 'LAUDO' in df.columns:
        # Conta quantas vezes cada diagnostico aparece
        contagem_classes = df['LAUDO'].value_counts()
        print(contagem_classes.to_string())
        
        num_classes = len(contagem_classes)
        print(f"\nTotal de classes unicas identificadas: {num_classes}")
    else:
        print("ERRO: A coluna LAUDO nao foi encontrada no dataset limpo.")

    # --- BLOCO 3: ANALISE DE DADOS FALTANTES NAS VARIAVEIS CLINICAS ---
    print("\n" + "-" * 40)
    print("SITUACAO DOS DADOS CLINICOS SELECIONADOS:")
    print("-" * 40)
    
    colunas_clinicas = ['IDADE', 'SEXO', 'COR', 'TABAGISTA', 'ETILISTA', 'EXPOSIÇÃO']
    for col in colunas_clinicas:
        if col in df.columns:
            # Mostra alguns valores unicos presentes para entendermos a padronizacao necessaria
            valores_amostra = df[col].dropna().unique()[:5]
            print(f"-> Coluna '{col}' | Exemplo de conteudos: {valores_amostra}")

if __name__ == '__main__':
    main()