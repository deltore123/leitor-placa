import psycopg2

def conectar():
    conexao = psycopg2.connect(
        host= "localhost", 
        port=5432,
        database= 'condominio',
        user='postgres',
        password= "1133"
    )
    return conexao