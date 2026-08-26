document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewImage = document.getElementById('preview-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading-state');
    const resultsContainer = document.getElementById('results-grid');
    
    // Read CSRF Token from meta tag
    const csrfMeta = document.querySelector('meta[name="csrf-token"]');
    const csrfToken = csrfMeta ? csrfMeta.getAttribute('content') : '';

    const ALLOWED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];
    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB limit
    
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
        // Strict MIME validation
        if (!file || !ALLOWED_MIME_TYPES.includes(file.type.toLowerCase())) {
            displayError('ERR: INVALID_FILE_TYPE (PNG/JPG ONLY)');
            return;
        }

        // File size validation (DoS / memory exhaustion prevention)
        if (file.size > MAX_FILE_SIZE) {
            displayError('ERR: FILE_SIZE_EXCEEDS_10MB_LIMIT');
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
        resultsContainer.textContent = ''; // Safely clear previous results

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const headers = {};
            if (csrfToken && csrfToken !== '{{CSRF_TOKEN}}') {
                headers['X-CSRF-Token'] = csrfToken;
            }

            const response = await fetch('/reconstruct', {
                method: 'POST',
                headers: headers,
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData && errData.error ? errData.error : `ERR_HTTP_${response.status}`);
            }
            
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            // Render Results Safely
            renderResults(data.results);

        } catch (error) {
            console.error(error);
            displayError(error.message || 'SYS_ERR // ANALYSIS_FAILED');
        } finally {
            // UI State: Done
            analyzeBtn.disabled = false;
            loadingState.classList.add('hidden');
        }
    });

    function displayError(message) {
        resultsContainer.textContent = '';
        
        const errorRow = document.createElement('div');
        errorRow.className = 'result-row';
        errorRow.style.borderColor = '#ef4444';

        const errorData = document.createElement('div');
        errorData.className = 'result-data';

        const errorLabel = document.createElement('span');
        errorLabel.className = 'mono-label';
        errorLabel.style.color = '#ef4444';
        errorLabel.textContent = 'SYS_ERR // ANALYSIS_FAILED';

        const errorVal = document.createElement('span');
        errorVal.className = 'data-value';
        errorVal.textContent = message; // Safe from XSS injection

        errorData.appendChild(errorLabel);
        errorData.appendChild(errorVal);
        errorRow.appendChild(errorData);
        resultsContainer.appendChild(errorRow);
    }

    function renderResults(results) {
        resultsContainer.textContent = '';
        
        if (!results || results.length === 0) {
            displayError('ERR: NO_RECONSTRUCTIONS_GENERATED');
            return;
        }

        results.forEach((result) => {
            const card = document.createElement('div');
            card.className = 'result-card';

            const imageContainer = document.createElement('div');
            imageContainer.className = 'result-image-container';

            const img = document.createElement('img');
            img.className = 'result-image';
            img.alt = 'Reconstruction';
            // Ensure data URL structure is respected
            if (typeof result.image_data === 'string' && result.image_data.startsWith('data:image/')) {
                img.src = result.image_data;
            }
            imageContainer.appendChild(img);

            const resultData = document.createElement('div');
            resultData.className = 'result-data';

            // Group: Rank
            const rankGroup = createDataGroup('RANK', `0${result.rank}`);
            // Group: Loss Metric
            const scoreVal = isNaN(parseFloat(result.score)) ? 'N/A' : parseFloat(result.score).toFixed(4);
            const scoreGroup = createDataGroup('FAN_MORPHOLOGICAL_LOSS', scoreVal, 'var(--accent)');
            // Group: Resolution
            const resGroup = createDataGroup('RESOLUTION', '256x256');

            resultData.appendChild(rankGroup);
            resultData.appendChild(scoreGroup);
            resultData.appendChild(resGroup);

            card.appendChild(imageContainer);
            card.appendChild(resultData);
            resultsContainer.appendChild(card);
        });
    }

    function createDataGroup(label, value, valueColor = null) {
        const group = document.createElement('div');
        group.className = 'data-group';

        const labelSpan = document.createElement('span');
        labelSpan.className = 'mono-label';
        labelSpan.textContent = label;

        const valSpan = document.createElement('span');
        valSpan.className = 'data-value';
        valSpan.textContent = value;
        if (valueColor) {
            valSpan.style.color = valueColor;
        }

        group.appendChild(labelSpan);
        group.appendChild(valSpan);
        return group;
    }
});
