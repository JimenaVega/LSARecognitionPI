const videoElement = document.getElementById('video');
const recordButton = document.getElementById('recordButton');
const clearHistoryButton = document.getElementById('clearHistoryButton');
const countdownElement = document.getElementById('countdown');
const countdownOverlay = document.getElementById('countdownOverlay');
const historyList = document.getElementById('historyList');
const statusIndicator = document.getElementById('statusIndicator');
const tooltip = document.getElementById('tooltip');

const options = { mimeType: 'video/webm' };
let mediaRecorder;
let countdownInterval;
const MAX_HISTORY_SIZE = 10;
let historyQueue = [];

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    checkAPIStatus(); // Verificar el estado de la API al cargar la página
});

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        videoElement.srcObject = stream;
        mediaRecorder = new MediaRecorder(stream, options);
        mediaRecorder.ondataavailable = function(event) {
            const formData = new FormData();
            formData.append('video', event.data);
            fetch('https://192.168.0.21:443/api/predict/', {
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

recordButton.addEventListener('click', () => {
    recordButton.disabled = true;
    clearHistoryButton.disabled = true;
    startCountdown();
});

clearHistoryButton.addEventListener('click', () => {
    clearHistoryButton.disabled = true;
    recordButton.disabled = true;
    clearHistory();
    clearHistoryButton.disabled = false;
    recordButton.disabled = false;
});

function startCountdown() {
    let countdown = 3;
    countdownElement.textContent = countdown;
    countdownOverlay.classList.add('visible');
    countdownInterval = setInterval(() => {
        countdown--;
        if (countdown > 0) {
            countdownElement.textContent = countdown;
        } else {
            clearInterval(countdownInterval);
            countdownOverlay.classList.remove('visible');
            startRecording();
        }
    }, 1000);
}

function startRecording() {
    mediaRecorder.start();
    setTimeout(() => {
        mediaRecorder.stop();
        countdownElement.textContent = "Procesando video...";
        countdownOverlay.classList.add('visible');
    }, 2500);
}

function addToHistory(sign) {
    historyQueue.unshift(sign);
    if (historyQueue.length > MAX_HISTORY_SIZE) {
        historyQueue.pop();
    }
    updateHistoryList();
    saveHistory();
    countdownOverlay.classList.remove('visible');
    recordButton.disabled = false;
    clearHistoryButton.disabled = false;
}

function updateHistoryList() {
    historyList.innerHTML = '';
    historyQueue.forEach((sign, index) => {
        const listItem = document.createElement('li');
        listItem.textContent = sign;
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
    } else {
        clearHistoryButton.disabled = false;
    }
}

function clearHistory() {
    historyQueue = [];
    updateHistoryList();
    localStorage.removeItem('history');
}

function checkAPIStatus() {
    fetch('https://192.168.0.21:443/api/status/')
        .then(response => {
            if (response.ok) {
                statusIndicator.style.backgroundColor = 'green';
                tooltip.textContent = 'Conectado';
            } else {
                statusIndicator.style.backgroundColor = 'red';
                tooltip.textContent = 'Desconectado';
            }
        })
        .catch(error => {
            console.error('Error al verificar el estado de la API:', error);
            statusIndicator.style.backgroundColor = 'red';
            tooltip.textContent = 'Desconectado';
        });
}

setInterval(checkAPIStatus, 10000);
