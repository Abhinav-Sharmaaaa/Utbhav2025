// Camera elements
const video = document.getElementById('camera');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const alertsContainer = document.getElementById('alerts-container');

// Camera stream reference
let stream = null;

// Initialize camera
async function initCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
        
        // Set canvas size to match video
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
        
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        // Start detection when camera is ready
        video.play();
        requestAnimationFrame(detectFrame);
    } catch (err) {
        console.error('Camera error:', err);
        showAlert('Camera access denied. Please enable camera permissions.', 'error');
    }
}

// Detection loop
function detectFrame() {
    if (video.paused || video.ended) return;
    
    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // TODO: Implement MediaPipe Pose detection
    // detectPoses();
    
    // TODO: Implement YOLO/YuNet object detection
    // detectObjects();
    
    // Temporary test detections
    if (Math.random() > 0.95) {
        // Simulate fall detection
        drawBoundingBox(100, 100, 200, 300, 'Fall detected', 'danger');
        showAlert('Warning: Possible fall detected!', 'danger');
    }
    
    if (Math.random() > 0.97) {
        // Simulate weapon detection
        drawBoundingBox(300, 200, 150, 150, 'Weapon detected', 'danger');
        showAlert('Danger: Weapon detected!', 'danger');
    }
    
    requestAnimationFrame(detectFrame);
}

// Draw bounding boxes with labels
function drawBoundingBox(x, y, width, height, label, type = 'warning') {
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.strokeStyle = type === 'danger' ? 'red' : 'orange';
    ctx.rect(x, y, width, height);
    ctx.stroke();
    
    // Draw label background
    ctx.fillStyle = type === 'danger' ? 'rgba(255,0,0,0.7)' : 'rgba(255,165,0,0.7)';
    ctx.fillRect(x, y - 20, ctx.measureText(label).width + 10, 20);
    
    // Draw label text
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.fillText(label, x + 5, y - 5);
}

// Draw pose keypoints
function drawKeypoints(keypoints) {
    keypoints.forEach(point => {
        const { x, y } = point;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'cyan';
        ctx.fill();
    });
}

// Alert system with flashing effect
function showAlert(message, type = 'warning') {
    const alert = document.createElement('div');
    alert.className = `alert ${type}`;
    alert.textContent = message;
    alertsContainer.prepend(alert);
    
    // Add flashing effect for danger alerts
    if (type === 'danger') {
        alert.style.animation = 'flash 1s infinite';
        
        // Add canvas overlay effect
        ctx.fillStyle = 'rgba(255,0,0,0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        setTimeout(() => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }, 500);
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Button handlers
startBtn.addEventListener('click', initCamera);
stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
});

// Initial setup
stopBtn.disabled = true;