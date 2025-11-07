
const btnPictogram = document.getElementById('btn-pictogram');
const btnAudio = document.getElementById('btn-audio');
const btnText = document.getElementById('btn-text');

btnPictogram.addEventListener('click', () => generateStory());
btnAudio.addEventListener('click', () => generateVoice());
btnText.addEventListener('click', () => generatePictogram());

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

        const data = await response.json();
        document.getElementById("output").textContent = data.text;
      } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = "Error al generar la historia.";
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

        const data = await response.json();
        document.getElementById("output").textContent = data.audio_path;
      } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = "Error al generar la voz.";
      }
}

async function generatePictogram() {
    const prompt = document.querySelector(".main-card-textarea").value;
    const requestBody = {
        "text": prompt
      };

      try {
        const response = await fetch("http://127.0.0.1:8000/pictogram/generate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(requestBody)
        });

        const data = await response.json();
        document.getElementById("output").textContent = data.pictogram_data;
      } catch (error) {
        console.error("Error:", error);
        document.getElementById("output").textContent = "Error al generar los pictogramas.";
      }
}
