import cv2
from deepface import DeepFace


captura = cv2.VideoCapture(0)
referencia_img = cv2.imread("usuarios/lucas.jpg")
if referencia_img is None:
    print("Erro: Imagem de referência não encontrada. Verifique o caminho: img/reference.jpg")
    exit()

# Variáveis para controle de estado e contagem
face_match = False
counter = 0
confidence = 0.0

# Pré-computa o embedding da imagem de referência para melhor performance
try:
    reference_embedding = DeepFace.represent(
        reference_img,
        model_name="Facenet512",      # Modelo de alta precisão para reconhecimento facial
        detector_backend="mtcnn",     # Detector de faces mais preciso
        enforce_detection=False       # Permite continuar mesmo se não detectar rostos
    )[0]["embedding"]
except Exception as e:
    print("Erro ao processar imagem de referência:", e)
    exit()

# Função para verificar se há correspondência facial no frame atual
def check_face(frame):
    global face_match, confidence
    try:
        # Extrai embedding facial do frame atual
        embedding_obj = DeepFace.represent(
            frame,
            model_name="Facenet512",
            detector_backend="mtcnn",
            enforce_detection=False
        )  
        # Se um rosto foi detectado, compara com a referência
        if embedding_obj:
            # Realiza a verificação facial comparando os embeddings
            result = DeepFace.verify(
                frame,
                reference_img,
                model_name="Facenet512",
                detector_backend="mtcnn",
                enforce_detection=False
            )
            face_match = result["verified"]
# Normaliza confiança em % (apenas para exibição)
            threshold = 0.7
            dist = result["distance"]
            confidence = max(0, (threshold - dist) / threshold)  # varia de 0 a 1
  # Calcula confiança baseada na distância
    except Exception as e:
        print("Erro na detecção facial:", e)
        face_match = False

# Loop principal de processamento de vídeo
while True:
    # Captura frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam")
        break
    # Processa o frame a cada 15 quadros
    if counter % 15 == 0:
        check_face(frame.copy())
    counter += 1
    # Exibe o resultado na tela
    status = "CORRESPONDENCIA" if face_match else "SEM CORRESPONDENCIA"
    color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, f"{status} ({confidence:.2%})", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # Mostra o frame processado
    cv2.imshow("Reconhecimento Facial", frame)
    # Encerra o loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos da câmera e fecha janelas
cap.release()
cv2.destroyAllWindows()

