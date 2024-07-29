const videoElement = document.getElementById('video');
const recordButton = document.getElementById('recordButton');
const signElement = document.getElementById('sign');
const countdownElement = document.getElementById('countdown');
const historyList = document.getElementById('historyList');

const options = { mimeType: 'video/webm' };
let mediaRecorder;
let countdownInterval;
const MAX_HISTORY_SIZE = 10; // Capacidad máxima del historial
let historyQueue = []; // Array para almacenar el historial

// Solicitar acceso a la cámara web
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;

        // Configurar MediaRecorder dentro del then para asegurar que mediaRecorder esté definido
        mediaRecorder = new MediaRecorder(stream, options);

        // Establecer ondataavailable después de crear mediaRecorder
        mediaRecorder.ondataavailable = function(event) {
            const formData = new FormData();
            formData.append('video', event.data);

            fetch('http://127.0.0.1:8000/items/', {
                method: 'POST',
                body: formData
            })
               .then(response => response.json())
               .then(response => {
                console.log(JSON.stringify(response));
                if (response.sign) {
                    signElement.textContent = response.sign;
                    addToHistory(response.sign);
                } else {
                    signElement.textContent = 'No sign received';
                }
            });
        };
    })
    .catch(error => console.error('Error al acceder a la cámara web:', error));

// Manejar clic en el botón de grabación
recordButton.addEventListener('click', startCountdown);

function startCountdown() {
    let countdown = 3;
    countdownElement.textContent = countdown;

    countdownInterval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            countdownElement.textContent = countdown;
        } else {
            clearInterval(countdownInterval);
            countdownElement.textContent = "Recording...";
            startRecording();
        }
    }, 1000);
}

function startRecording() {
    // Iniciar la grabación
    mediaRecorder.start();

    // Grabar durante 2 segundos
    setTimeout(() => {
        mediaRecorder.stop();
        countdownElement.textContent = "Not recording";
    }, 2000);
}

function addToHistory(sign) {
    // Añadir el nuevo signo al historial
    historyQueue.unshift(sign); // Añade al principio del array

    // Mantener solo los últimos MAX_HISTORY_SIZE elementos
    if (historyQueue.length > MAX_HISTORY_SIZE) {
        historyQueue.pop(); // Elimina el último elemento
    }

    // Actualizar la vista del historial
    updateHistoryList();
}

function updateHistoryList() {
    historyList.innerHTML = ''; // Limpiar el historial actual
    historyQueue.forEach(sign => {
        const listItem = document.createElement('li');
        listItem.textContent = sign;
        historyList.appendChild(listItem);
    });
}
