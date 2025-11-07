// Espera a que todo el contenido HTML esté cargado
document.addEventListener('DOMContentLoaded', () => {

    // --- NUEVO: Lógica de la Pantalla de Bienvenida ---
    const welcomeScreen = document.getElementById('welcome-screen');
    const appScreen = document.getElementById('app-screen');
    const startAppButton = document.getElementById('start-app-button');

    // Listener para empezar la app
    startAppButton.addEventListener('click', () => {
        // Ocultar bienvenida
        welcomeScreen.style.display = 'none';
        
        // Mostrar app con una pequeña animación de entrada
        appScreen.style.display = 'block';
        appScreen.style.animation = 'fadeIn 0.5s ease-out';
    });


    // --- ======== LÓGICA DE LA APLICACIÓN EXISTENTE ======== ---

    // --- Variables Globales ---
    let historyDB = {
        pictogram: [],
        audio: [],
        text: []
    };
    const storageKey = 'touieStoryHistory';

    // --- Selectores de Elementos ---
    const storyInput = document.getElementById('story-input');
    const outputArea = document.getElementById('story-output');
    
    // Botones de generación
    const btnPictogram = document.getElementById('btn-pictogram');
    const btnAudio = document.getElementById('btn-audio');
    const btnText = document.getElementById('btn-text');

    // Pestañas
    const tabCreator = document.getElementById('tab-creator');
    const tabHistory = document.getElementById('tab-history');
    
    // Paneles
    const creatorPane = document.getElementById('creator-pane');
    const historyPane = document.getElementById('history-pane');

    // Contenedores de listas de historial
    const historyPictogramList = document.getElementById('history-pictogram');
    const historyAudioList = document.getElementById('history-audio');
    const historyTextList = document.getElementById('history-text');

    // --- Lógica de Pestañas ---
    tabCreator.addEventListener('click', () => {
        // Activar Pestaña
        tabCreator.classList.add('active');
        tabHistory.classList.remove('active');
        // Mostrar Panel
        creatorPane.classList.add('active');
        historyPane.classList.remove('active');
    });

    tabHistory.addEventListener('click', () => {
        // Activar Pestaña
        tabHistory.classList.add('active');
        tabCreator.classList.remove('active');
        // Mostrar Panel
        historyPane.classList.add('active');
        creatorPane.classList.remove('active');
        
        // Actualizar la vista del historial cada vez que se abre
        renderHistoryUI();
    });
    // --- Lógica de Generación de Historias ---
    btnPictogram.addEventListener('click', () => generateStory('pictogram'));
    btnAudio.addEventListener('click', () => generateStory('audio'));
    btnText.addEventListener('click', () => generateStory('text'));

    function generateStory(format) {
        const inputText = storyInput.value;
       
        if (inputText.trim() === "") {
            outputArea.innerHTML = `
                <div class="placeholder">
                    <img src="/CiberPaz-2025/frontend/resources/right.png" alt="Touie señalando" class="mascot-waiting" style="width: 80px;">
                    <p style="color: var(--rojo-destacado);">¡Ups! Primero escribe tu idea arriba.</p>
                </div>
            `;
            return;
        }

        showLoadingState();

        setTimeout(() => {
            // Mostrar el resultado en la UI
            showResult(format, inputText);
            
            // Crear el objeto de datos para guardar
            const storyData = {
                id: Date.now(), // ID único basado en la fecha
                date: new Date().toLocaleDateString('es-ES'),
                input: inputText
            };
            
            // Guardar en la "base de datos" y en localStorage
            saveHistory(format, storyData);

        }, 2000); // 2 segundos de simulación
    }

    function showLoadingState() {
        outputArea.innerHTML = `
            <div class="loading">
                <img src="/CiberPaz-2025/frontend/resources/thinking.png" alt="Touie pensando" class="mascot-thinking">
                <p>¡Touie está creando tu historia!</p>
            </div>
        `;
    }

    function showResult(format, text) {
        let htmlResult = '';
        switch (format) {
            case 'pictogram':
                htmlResult = `
                    <h3>Tu Historia en Pictogramas</h3>
                    <p style="font-size: 28px;">👦 ➡️ 🏠  gặp 🐶 ➡️ 🌳 🌙</p>
                    <p>(Simulación de pictogramas basada en: "${text}")</p>
                `;
                break;
            case 'audio':
                htmlResult = `
                    <h3>Tu Historia en Audio</h3>
                    <p>¡Escucha tu increíble historia!</p>
                    <audio controls style="width: 100%; margin-top: 10px;">
                        <source src="" type="audio/mpeg">Tu navegador no soporta audio.
                    </audio>
                    <p>(Simulación de audio de: "${text}")</p>
                `;
                break;
            case 'text':
                htmlResult = `
                    <h3>Tu Historia en Texto</h3>
                    <p>Había una vez, en un lugar muy lejano, ${text}. Y todos vivieron felices para siempre.</p>
                `;
                break;
        }
        outputArea.innerHTML = htmlResult;
    }

    // --- Lógica de Historial ---

    function saveHistory(format, storyData) {
        // Añade la nueva historia al array correspondiente (al principio)
        historyDB[format].unshift(storyData);
        
        // Limita el historial a 10 items por categoría (opcional)
        if (historyDB[format].length > 10) {
            historyDB[format].pop();
        }

        // Guarda en localStorage
        localStorage.setItem(storageKey, JSON.stringify(historyDB));
        
        // Actualiza la UI del historial (si el usuario está en esa pestaña)
        if (historyPane.classList.contains('active')) {
            renderHistoryUI();
        }
    }

    function loadHistoryFromStorage() {
        const storedHistory = localStorage.getItem(storageKey);
        if (storedHistory) {
            historyDB = JSON.parse(storedHistory);
            // Asegurarse de que las propiedades existen
            if (!historyDB.pictogram) historyDB.pictogram = [];
            if (!historyDB.audio) historyDB.audio = [];
            if (!historyDB.text) historyDB.text = [];
        }
        // Renderizar por primera vez (aunque la pestaña esté oculta)
        renderHistoryUI();
    }

    function renderHistoryUI() {
        // 1. Renderizar Pictogramas
        renderList(historyDB.pictogram, historyPictogramList, "Aún no hay historias de pictogramas.");
        
        // 2. Renderizar Audio
        renderList(historyDB.audio, historyAudioList, "Aún no hay historias de audio.");

        // 3. Renderizar Texto
        renderList(historyDB.text, historyTextList, "Aún no hay historias de texto.");
    }

    // Función auxiliar para renderizar una lista
    function renderList(items, listElement, emptyMessage) {
        listElement.innerHTML = ''; // Limpiar la lista
        
        if (items.length === 0) {
            listElement.innerHTML = `<p class="empty-state">${emptyMessage}</p>`;
        } else {
            items.forEach(item => {
                const historyItemHTML = `
                    <div class="history-item" data-id="${item.id}">
                        <small>${item.date}</small>
                        <p><strong>Idea:</strong> ${item.input}</p>
                    </div>
                `;
                listElement.innerHTML += historyItemHTML;
            });
        }
    }

    // --- Inicialización ---
    // Cargar el historial guardado cuando la página se carga por primera vez
    loadHistoryFromStorage();
});