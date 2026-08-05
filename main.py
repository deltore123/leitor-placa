import psycopg2
from render import ler_placa
from database import conectar

try:
    conexao = conectar()
    print("✅ Conectado ao PostgreSQL!")
    conexao.close()

except Exception as erro:
    print("Erro:", erro)

placa = ler_placa(r'C:\Users\gabri\Documents\projetos\leitor-placa-main\leitor-placa\f1.jpg')
print (placa)
