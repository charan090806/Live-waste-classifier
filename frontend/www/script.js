document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const toggleBtn = document.getElementById('toggleCamera');
    const statusText = document.getElementById('apiStatusText');
    const statusDot = document.getElementById('apiStatusDot');
    
    // State
    let isCameraActive = true;
    
    // Listen for connection status changes
    document.addEventListener('api-status', (e) => {
        const status = e.detail;
        if (status === 'connected') {
            statusDot.className = 'dot active';
            statusText.textContent = 'System Active';
        } else if (status === 'error') {
            statusDot.className = 'dot error';
            statusText.textContent = 'API Offline or Model loading...';
        } else {
            statusDot.className = 'dot';
            statusText.textContent = 'Connecting...';
        }
    });

    // Toggle Camera
    toggleBtn.addEventListener('click', () => {
        isCameraActive = !isCameraActive;
        const event = new CustomEvent('toggle-camera', { detail: isCameraActive });
        document.dispatchEvent(event);
        
        if (isCameraActive) {
            toggleBtn.innerHTML = '<span class="icon">📷</span> Stop Camera';
            toggleBtn.className = 'btn primary';
            toggleBtn.style.background = ''; // reset to default css
        } else {
            toggleBtn.innerHTML = '<span class="icon">📷</span> Start Camera';
            toggleBtn.className = 'btn';
            toggleBtn.style.background = '#555';
            statusDot.className = 'dot';
            statusText.textContent = 'Camera Paused';
        }
    });
});
