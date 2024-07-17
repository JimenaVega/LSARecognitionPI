const videoElement = document.getElementById('video');
const recordButton = document.getElementById('recordButton');

const options = { mimeType: 'video/webm' };
let mediaRecorder;
let chunks = [];

// Solicitar acceso a la cámara web
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;

        // Configurar MediaRecorder dentro del then para asegurar que mediaRecorder esté definido
        mediaRecorder = new MediaRecorder(stream, options);

        // Establecer ondataavailable después de crear mediaRecorder
        mediaRecorder.ondataavailable = function(event) {
            chunks.push(event.data);
        };
    })
    .catch(error => console.error('Error al acceder a la cámara web:', error));

// Manejar clic en el botón de grabación
recordButton.addEventListener('click', startRecording);

function startRecording() {
    // Iniciar la grabación
    mediaRecorder.start();

    // Grabar durante 5 segundos
    setTimeout(() => {
        mediaRecorder.stop();

        // Enviar el video a la API
        const videoBlob = new Blob(chunks, options);
        const formData = new FormData();
        formData.append('video', videoBlob);

        // Reemplazar "YOUR_API_URL" con la URL de tu API
        fetch('http://127.0.0.1:8000/items/', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ "name": "test" })
        })
           .then(response => response.json())
           .then(response => console.log(JSON.stringify(response)))
    }, 1000);
}
