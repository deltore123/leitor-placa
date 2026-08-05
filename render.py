import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

# =========================================================
# CONFIGURAÇÕES
# =========================================================
LARGURA_PADRAO = 600  # redimensiona a imagem para essa largura (melhora consistência)

# Formatos de placa brasileira:
#   Antiga:   ABC1234   -> 3 letras + 4 números
#   Mercosul: ABC1D23   -> 3 letras + 1 número + 1 letra + 2 números
REGEX_PLACA_ANTIGA = re.compile(r'^[A-Z]{3}[0-9]{4}$')
REGEX_PLACA_MERCOSUL = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')

# Cache do leitor EasyOCR — carregar o modelo é lento, então só faz isso 1 vez
_leitor_ocr = None


def _get_leitor_ocr():
    global _leitor_ocr
    if _leitor_ocr is None:
        _leitor_ocr = easyocr.Reader(['pt'])
    return _leitor_ocr


def redimensionar(img, largura):
    h, w = img.shape[:2]
    escala = largura / w
    return cv2.resize(img, (largura, int(h * escala))), escala


def auto_canny(imagem, sigma=0.33):
    """Calcula thresholds do Canny automaticamente com base na mediana da imagem."""
    mediana = np.median(imagem)
    inferior = int(max(0, (1.0 - sigma) * mediana))
    superior = int(min(255, (1.0 + sigma) * mediana))
    return cv2.Canny(imagem, inferior, superior)


def encontrar_placa(img_bgr):
    """
    Retorna (x, y, w, h) da região candidata a placa, ou None se não encontrar.
    Filtra contornos por posição, proporção largura/altura típica de placa,
    faixa de área e — o critério mais confiável — brilho/contraste (placa
    brasileira é clara com texto preto, então tem brilho médio e desvio
    padrão altos comparado a outras regiões do carro).
    """
    H, W = img_bgr.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = auto_canny(bfilter)
    edged = cv2.dilate(edged, None, iterations=1)  # fecha pequenas quebras nas bordas

    contornos, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue

        aspecto = w / float(h)
        area = w * h

        posicao_ok = y > H * 0.40
        aspecto_ok = 2.2 <= aspecto <= 4.5
        area_ok = (0.003 * W * H) <= area <= (0.08 * W * H)

        if posicao_ok and aspecto_ok and area_ok:
            regiao = gray[y:y + h, x:x + w]
            brilho = regiao.mean()
            contraste = regiao.std()
            candidatos.append((x, y, w, h, aspecto, brilho, contraste))

    if not candidatos:
        return None, gray, edged

    def score(c):
        _, _, _, _, aspecto, brilho, contraste = c
        return -(brilho / 255) - (contraste / 255) * 0.5 + abs(aspecto - 3.1) * 0.1

    candidatos.sort(key=score)
    x, y, w, h, *_ = candidatos[0]
    return (x, y, w, h), gray, edged


def _normalizar_texto(texto):
    """Deixa só letras/números maiúsculos, removendo espaços, hífens, pontos etc."""
    return re.sub(r'[^A-Z0-9]', '', texto.upper())


def _escolher_melhor_leitura(resultados_ocr):
    """
    resultados_ocr: lista de (bbox, texto, confianca) vinda do easyocr.
    Escolhe o melhor candidato a placa: prioriza textos que batem com o
    formato de placa brasileira (antiga ou Mercosul); se nenhum bater
    exatamente, cai pra maior confiança / maior texto concatenado.
    """
    if not resultados_ocr:
        return None

    candidatos_validos = []
    for _, texto, confianca in resultados_ocr:
        limpo = _normalizar_texto(texto)
        if REGEX_PLACA_ANTIGA.match(limpo) or REGEX_PLACA_MERCOSUL.match(limpo):
            candidatos_validos.append((limpo, confianca))

    if candidatos_validos:
        candidatos_validos.sort(key=lambda c: c[1], reverse=True)
        return candidatos_validos[0][0]

    # Nenhum resultado bateu exatamente com o formato -> concatena tudo que
    # o OCR leu (ordenado por posição horizontal) e tenta usar como fallback
    resultados_ordenados = sorted(resultados_ocr, key=lambda r: r[0][0][0])
    texto_concatenado = _normalizar_texto(
        ''.join(texto for _, texto, _ in resultados_ordenados)
    )
    return texto_concatenado or None


def ler_placa(caminho_imagem, mostrar_debug=False):
    """
    Recebe o caminho de uma imagem de carro e retorna o texto da placa (str).

    Retorna None se:
      - a imagem não puder ser carregada
      - nenhuma região candidata a placa for encontrada
      - o OCR não conseguir ler nada na região encontrada

    Parâmetros:
      caminho_imagem (str): caminho para o arquivo de imagem
      mostrar_debug (bool): se True, exibe as etapas intermediárias (matplotlib)

    Exemplo:
      >>> placa = ler_placa("carro.png")
      >>> print(placa)
      'R102A19'
    """
    img = cv2.imread(caminho_imagem)
    if img is None:
        print(f"Imagem não encontrada: {caminho_imagem}")
        return None

    img, _ = redimensionar(img, LARGURA_PADRAO)

    resultado_deteccao, gray, edged = encontrar_placa(img)

    if mostrar_debug:
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Imagem redimensionada")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(edged, cmap="gray")
        plt.title("Bordas (Canny)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        vis = img.copy()
        if resultado_deteccao:
            x, y, w, h = resultado_deteccao
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Região candidata")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if resultado_deteccao is None:
        print("Nenhuma placa encontrada!")
        return None

    x, y, w, h = resultado_deteccao

    # Pequena margem ao redor da placa ajuda o OCR a não cortar caracteres
    margem = 4
    y1 = max(0, y - margem)
    y2 = min(gray.shape[0], y + h + margem)
    x1 = max(0, x - margem)
    x2 = min(gray.shape[1], x + w + margem)

    cropped_image = gray[y1:y2, x1:x2]

    # Aumenta a imagem da placa — melhora bastante a precisão do EasyOCR
    cropped_image = cv2.resize(
        cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
    )

    if mostrar_debug:
        plt.figure(figsize=(8, 4))
        plt.imshow(cropped_image, cmap="gray")
        plt.title("Placa recortada (ampliada para OCR)")
        plt.axis("off")
        plt.show()

    leitor = _get_leitor_ocr()
    resultados_ocr = leitor.readtext(cropped_image)

    if mostrar_debug:
        print("\nResultado OCR bruto:")
        for bbox, texto, confianca in resultados_ocr:
            print(f"  '{texto}'  |  confiança: {confianca:.2f}")

    return _escolher_melhor_leitura(resultados_ocr)


if __name__ == "__main__":
    CAMINHO_IMAGEM = r"C:\Users\gabri\Documents\projetos\leitor-placa-main\leitor-placa\image.png"

    placa = ler_placa(CAMINHO_IMAGEM, mostrar_debug=True)

    if placa:
        print(f"\nPlaca lida: {placa}")
    else:
        print("\nNão foi possível ler a placa.")