// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const generateBtn = document.getElementById('generateBtn');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const changeImage = document.getElementById('changeImage');
const toastContainer = document.getElementById('toastContainer');

// Supported file types
const validFileTypes = ['image/jpeg', 'image/png', 'image/jpg'];
const maxFileSize = 5 * 1024 * 1024; // 5MB

// Event Listeners
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
dropZone.addEventListener('dragover', handleDragOver);
dropZone.addEventListener('dragleave', handleDragLeave);
dropZone.addEventListener('drop', handleDrop);
changeImage.addEventListener('click', resetFileInput);
generateBtn.addEventListener('click', processImage);

// Handle file selection via input
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) validateAndPreviewFile(file);
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('highlight');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('highlight');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('highlight');
    
    const file = e.dataTransfer.files[0];
    if (file) validateAndPreviewFile(file);
}

// Validate and preview file
function validateAndPreviewFile(file) {
    // Check file type
    if (!validFileTypes.includes(file.type)) {
        showToast('Invalid file type. Please upload a JPG or PNG image.', 'error');
        return;
    }

    // Check file size
    if (file.size > maxFileSize) {
        showToast('File is too large. Max size is 5MB.', 'error');
        return;
    }

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        previewContainer.classList.remove('hidden');
        generateBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Reset file input
function resetFileInput() {
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    generateBtn.disabled = true;
}

// Process image using local AI
async function processImage() {
    const file = fileInput.files[0];
    if (!file) return;

    showToast('Generating designs with local AI...', 'success');
    
    try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('prompt', 'professional design variation');
        
        const response = await fetch('http://localhost:5000/generate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        sessionStorage.setItem('aiResults', JSON.stringify(data));
        window.location.href = 'process.html';
        
    } catch (error) {
        showToast('AI generation failed. Is local server running?', 'error');
        console.error(error);
    }
}

// Show toast notification
function showToast(message, type) {
    const toast = document.createElement('div');
    toast.className = `toast px-4 py-2 rounded-lg shadow-md flex items-center ${
        type === 'error' ? 'bg-red-500' : 'bg-green-500'
    } text-white`;
    toast.innerHTML = `
        <i class="fas ${type === 'error' ? 'fa-exclamation-circle' : 'fa-check-circle'} mr-2"></i>
        ${message}
    `;
    toastContainer.appendChild(toast);
    
    // Remove toast after animation
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}