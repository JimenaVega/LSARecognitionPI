const videoElement = document.getElementById('video');
const recordButton = document.getElementById('recordButton');
const clearHistoryButton = document.getElementById('clearHistoryButton');
const countdownElement = document.getElementById('countdown');
const historyList = document.getElementById('historyList');

const options = { mimeType: 'video/webm' };
let mediaRecorder;
let countdownInterval;
const MAX_HISTORY_SIZE = 10; // Capacidad máxima del historial
let historyQueue = []; // Array para almacenar el historial

// Cargar el historial desde localStorage al iniciar
document.addEventListener('DOMContentLoaded', loadHistory);

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
                    addToHistory(response.sign);
                } else {
                    addToHistory('No hay seña recibida.');
                }
            });
        };
    })
    .catch(error => console.error('Error al acceder a la cámara web:', error));

// Manejar clic en el botón de grabación
recordButton.addEventListener('click', startCountdown);

// Manejar clic en el botón de limpiar historial
clearHistoryButton.addEventListener('click', clearHistory);

function startCountdown() {
    let countdown = 3;
    countdownElement.textContent = countdown;

    countdownInterval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            countdownElement.textContent = countdown;
        } else {
            clearInterval(countdownInterval);
            countdownElement.textContent = "Grabando...";
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
        countdownElement.textContent = "No se encuentra grabando.";
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
    saveHistory(); // Guardar el historial en localStorage
}

function updateHistoryList() {
    historyList.innerHTML = ''; // Limpiar el historial actual

    historyQueue.forEach((sign, index) => {
        const listItem = document.createElement('li');
        listItem.textContent = sign;

        // Resaltar el último elemento
        if (index === 0) {
            listItem.classList.add('highlight');
        }

        historyList.appendChild(listItem);
    });
}

function saveHistory() {
    localStorage.setItem('history', JSON.stringify(historyQueue));
}

function loadHistory() {
    const savedHistory = localStorage.getItem('history');
    if (savedHistory) {
        historyQueue = JSON.parse(savedHistory);
        updateHistoryList();
    }
}

function clearHistory() {
    historyQueue = []; // Vaciar el array del historial
    updateHistoryList(); // Actualizar la vista del historial
    localStorage.removeItem('history'); // Eliminar el historial guardado en localStorage
}
