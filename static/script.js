document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-grid');
    
    let currentFile = null;

    // Drag and Drop Logic
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('ERR: INVALID_MIME_TYPE');
            return;
        }
        
        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Analysis Logic
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        loadingState.classList.remove('hidden');
        resultsContainer.innerHTML = ''; // Clear previous results

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/reconstruct', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('ERR_NET_RESPONSE_INVALID');
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Render Results
            renderResults(data.results);

        } catch (error) {
            console.error(error);
            resultsContainer.innerHTML = `
                <div class="result-row" style="border-color: #ef4444;">
                    <div class="result-data">
                        <span class="mono-label" style="color: #ef4444;">SYS_ERR // ANALYSIS_FAILED</span>
                        <span class="data-value">${error.message}</span>
                    </div>
                </div>
            `;
        } finally {
            // UI State: Done
            analyzeBtn.disabled = false;
            loadingState.classList.add('hidden');
        }
    });

    function renderResults(results) {
        resultsContainer.innerHTML = '';
        
        results.forEach((result, index) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            // Format score to 4 decimal places
            const formattedScore = parseFloat(result.score).toFixed(4);
            
            card.innerHTML = `
                <div class="result-image-container">
                    <img src="${result.image_data}" class="result-image" alt="Reconstruction">
                </div>
                <div class="result-data">
                    <div class="data-group">
                        <span class="mono-label">RANK</span>
                        <span class="data-value">0${result.rank}</span>
                    </div>
                    <div class="data-group">
                        <span class="mono-label">FAN_MORPHOLOGICAL_LOSS</span>
                        <span class="data-value" style="color: var(--accent);">${formattedScore}</span>
                    </div>
                    <div class="data-group">
                        <span class="mono-label">RESOLUTION</span>
                        <span class="data-value">256x256</span>
                    </div>
                </div>
            `;
            
            resultsContainer.appendChild(card);
        });
    }
});
