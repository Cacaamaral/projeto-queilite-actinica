import pandas as pd
import numpy as np
import os
import glob

def main():
    # --- BLOCO 1: MAPEAMENTO DE DIRETORIOS E ARQUIVOS ---
    DIRETORIO_CODIGOS = os.path.dirname(os.path.abspath(__file__))
    DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGOS)
    DIRETORIO_FOTOS = os.path.join(DIRETORIO_RAIZ, 'fotos')
    
    ARQUIVO_BD = os.path.join(DIRETORIO_RAIZ, 'Banco de dados QA_incompleto.xlsx')
    ARQUIVO_SAIDA = os.path.join(DIRETORIO_CODIGOS, 'dataset_qa_limpo.csv')

    print("Iniciando processamento com foco na coluna REF...")

    # --- BLOCO 2: DETECCAO AUTOMATICA DA LINHA DE CABECALHO ---
    try:
        df_temp = pd.read_excel(ARQUIVO_BD, engine='openpyxl', header=None, nrows=10)
        
        linha_cabecalho = 0
        for indice, linha in df_temp.iterrows():
            valores_linha = [str(val).strip().upper() for val in linha.values]
            if 'REF' in valores_linha:
                linha_cabecalho = indice
                break
                
        df = pd.read_excel(ARQUIVO_BD, engine='openpyxl', header=linha_cabecalho)
        
    except Exception as e:
        print(f"ERRO ao tentar ler o arquivo Excel: {e}")
        return

    # --- BLOCO 3: PADRONIZACAO UNIVERSAL DE CABECALHOS ---
    df.columns = df.columns.astype(str).str.strip().str.upper().str.replace(' ', '_')

    # --- BLOCO 4: FILTRAGEM PELA COLUNA REF ---
    coluna_id = 'REF'
    
    if coluna_id not in df.columns:
        print(f"ERRO CRITICO: Coluna '{coluna_id}' nao encontrada apos a leitura.")
        return

    # Remove linhas vazias na coluna REF e garante que sejam numeros inteiros
    df = df.dropna(subset=[coluna_id])
    df[coluna_id] = pd.to_numeric(df[coluna_id], errors='coerce').fillna(0).astype(int)

    # --- BLOCO 5: MAPEAMENTO DAS IMAGENS (LATE FUSION PREP) ---
    caminhos_imagens = []
    ids_validos = []

    for index, row in df.iterrows():
        paciente_id = str(row[coluna_id])
        pasta_paciente = os.path.join(DIRETORIO_FOTOS, paciente_id)
        
        if os.path.exists(pasta_paciente):
            # Busca imagens suportadas dentro da pasta do paciente especifico
            imagens = glob.glob(os.path.join(pasta_paciente, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(pasta_paciente, '*.[jJ][pP][eE][gG]')) + \
                      glob.glob(os.path.join(pasta_paciente, '*.[pP][nN][gG]'))
            
            if len(imagens) > 0:
                caminhos_imagens.append(imagens[0])
                ids_validos.append(True)
            else:
                caminhos_imagens.append(np.nan)
                ids_validos.append(False)
        else:
            caminhos_imagens.append(np.nan)
            ids_validos.append(False)

    df['CAMINHO_IMAGEM'] = caminhos_imagens
    
    # Cria o dataframe final apenas com pacientes que possuem foto
    df_filtrado = df[ids_validos].copy()
    
    print(f"Total de registros originais na planilha: {len(df)}")
    print(f"Total de pacientes com foto localizada e vinculada: {len(df_filtrado)}")

    if len(df_filtrado) == 0:
        print("\nAVISO: Nenhuma foto foi vinculada. Verifique se as pastas dentro de 'QA/fotos/' estao nomeadas apenas com numeros (ex: '1001', '1002').")
        return

    # --- BLOCO 6: HIGIENIZACAO DINAMICA DE DADOS ---
    # Incluido 'string' para corrigir o aviso de descontinuacao do Pandas
    colunas_texto = df_filtrado.select_dtypes(include=['object', 'string']).columns
    for col in colunas_texto:
        df_filtrado[col] = df_filtrado[col].astype(str).str.strip().str.upper()
        df_filtrado[col] = df_filtrado[col].replace(['NAN', 'NONE', 'NULL', ''], np.nan)
        df_filtrado[col] = df_filtrado[col].fillna('NAO_INFORMADO')

    colunas_numericas = df_filtrado.select_dtypes(include=[np.number]).columns
    for col in colunas_numericas:
        if col != coluna_id: 
            mediana = df_filtrado[col].median()
            df_filtrado[col] = df_filtrado[col].fillna(mediana)

    # --- BLOCO 7: SELECAO DAS VARIAVEIS ALVO E SALVAMENTO ---
    # Para o treinamento, precisamos garantir que as variaveis criticas existam e estejam limpas
    variaveis_interesse = ['REF', 'IDADE', 'SEXO', 'COR', 'TABAGISTA', 'ETILISTA', 'EXPOSIÇÃO', 'LAUDO', 'CAMINHO_IMAGEM']
    
    # Filtra apenas as colunas que realmente importam para o modelo hibrido, ignorando as vazias
    colunas_finais = [col for col in variaveis_interesse if col in df_filtrado.columns]
    df_final = df_filtrado[colunas_finais]

    df_final.to_csv(ARQUIVO_SAIDA, index=False)
    print(f"\nProcessamento concluido com sucesso. Arquivo limpo salvo em:\n{ARQUIVO_SAIDA}")

if __name__ == '__main__':
    main()