const btnPictogram = document.getElementById('btn-pictogram');
const btnAudio = document.getElementById('btn-audio');
const btnText = document.getElementById('btn-text');

btnPictogram.addEventListener('click', () => generatePictogram());
btnAudio.addEventListener('click', () => generateVoice());
btnText.addEventListener('click', () => generateStory());

async function generateStory() {
    const prompt = document.querySelector(".main-card-textarea").value;
    
    const requestBody = {
        prompt: prompt,
        max_tokens: 120,
        tone: "calmo",
        complexity: "simple",
        sensory_friendly: true,
        story_type: "cotidiana",
        protagonist_name: ""
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/text/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById("output").textContent = data.text;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = `Error al generar la historia: ${error.message}`;
    }
}

async function generateVoice() {
    const prompt = document.querySelector(".main-card-textarea").value;
    
    const requestBody = {
        "text": prompt,
        "language": "es",
        "voice_speed": 1,
        "speaker_wav": "string"
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/voice/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        document.getElementById("output").textContent = data.audio_path;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = `Error al generar la voz: ${error.message}`;
    }
}

async function generatePictogram() {
    const prompt = document.querySelector(".main-card-textarea").value;
    
    // El endpoint /pictogram/from_prompt espera el mismo schema que /text/generate
    const requestBody = {
        prompt: prompt,
        max_tokens: 120,
        tone: "calmo",
        complexity: "simple",
        sensory_friendly: true,
        story_type: "cotidiana",
        protagonist_name: ""
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/pictogram/from_prompt", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Renderizar la historia y los pictogramas
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML = ""; // Limpiar contenido previo
        
        // Mostrar la historia generada
        if (data.story) {
            const storyElement = document.createElement("div");
            storyElement.className = "story-output";
            storyElement.innerHTML = `<h3>Historia:</h3><p>${data.story}</p>`;
            outputDiv.appendChild(storyElement);
        }
        
        // Mostrar los pictogramas
        if (data.pictograms && data.pictograms.items && data.pictograms.items.length > 0) {
            const pictogramsContainer = document.createElement("div");
            pictogramsContainer.className = "pictograms-container";
            pictogramsContainer.innerHTML = "<h3>Pictogramas:</h3>";
            
            const pictogramsGrid = document.createElement("div");
            pictogramsGrid.className = "pictograms-grid";
            pictogramsGrid.style.cssText = "display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; margin-top: 1rem;";
            
            data.pictograms.items.forEach(item => {
                const pictogramCard = document.createElement("div");
                pictogramCard.className = "pictogram-card";
                pictogramCard.style.cssText = "text-align: center; padding: 0.5rem; border: 1px solid #ddd; border-radius: 8px;";
                
                const img = document.createElement("img");
                img.src = item.image; // La imagen ya viene en formato data:image/png;base64,...
                img.alt = item.alt || item.label;
                img.style.cssText = "width: 100%; height: auto; border-radius: 4px;";
                
                const label = document.createElement("p");
                label.textContent = item.label;
                label.style.cssText = "margin-top: 0.5rem; font-weight: bold; font-size: 0.9rem;";
                
                pictogramCard.appendChild(img);
                pictogramCard.appendChild(label);
                pictogramsGrid.appendChild(pictogramCard);
            });
            
            pictogramsContainer.appendChild(pictogramsGrid);
            outputDiv.appendChild(pictogramsContainer);
        } else {
            const noImagesMsg = document.createElement("p");
            noImagesMsg.textContent = "No se generaron pictogramas.";
            outputDiv.appendChild(noImagesMsg);
        }
        
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = `Error al generar los pictogramas: ${error.message}`;
    }
}
