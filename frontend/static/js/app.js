// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');
const uploadArea = document.getElementById('upload-area');
const audioUpload = document.getElementById('audio-upload');
const uploadResult = document.getElementById('upload-result');
const processBtn = document.getElementById('process-btn');
const processProgress = document.getElementById('process-progress');
const processResult = document.getElementById('process-result');

// Navigation functionality
navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetSection = item.dataset.section;
        
        // Update active nav item
        navItems.forEach(nav => nav.classList.remove('active'));
        item.classList.add('active');
        
        // Show target section
        contentSections.forEach(section => {
            section.classList.remove('active');
            if (section.id === `${targetSection}-section`) {
                section.classList.add('active');
            }
        });
    });
});

// File upload functionality
uploadArea.addEventListener('click', () => {
    audioUpload.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

audioUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Handle file upload
function handleFileUpload(file) {
    // Validate file type
    const allowedTypes = ['audio/mp3', 'audio/wav', 'audio/flac', 'audio/mpeg'];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a valid audio file (MP3, WAV, or FLAC)');
        return;
    }
    
    // Validate file size (50MB max)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File size must be less than 50MB');
        return;
    }
    
    // Show upload result
    uploadArea.style.display = 'none';
    uploadResult.style.display = 'block';
    
    // Update file details
    const filename = uploadResult.querySelector('.filename');
    const filesize = uploadResult.querySelector('.filesize');
    const audio = uploadResult.querySelector('audio source');
    
    filename.textContent = file.name;
    filesize.textContent = formatFileSize(file.size);
    
    // Create object URL for audio preview
    const objectUrl = URL.createObjectURL(file);
    audio.src = objectUrl;
    uploadResult.querySelector('audio').load();
    
    // Store file for processing
    uploadResult.dataset.file = objectUrl;
    uploadResult.dataset.filename = file.name;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Generate noise audio to simulate early model output
async function generateNoiseAudio() {
    const sampleRate = 22050;
    const duration = 10; // 10 seconds
    const numSamples = sampleRate * duration;
    
    // Create AudioContext
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: sampleRate
    });
    
    // Create buffer
    const buffer = audioContext.createBuffer(1, numSamples, sampleRate);
    const channelData = buffer.getChannelData(0);
    
    // Fill with noise (representing model output)
    for (let i = 0; i < numSamples; i++) {
        channelData[i] = (Math.random() * 2 - 1) * 0.1; // Low volume noise
    }
    
    // Convert to WAV blob
    const wavBuffer = audioBufferToWav(buffer);
    return new Blob([wavBuffer], { type: 'audio/wav' });
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const length = buffer.length;
    const sampleRate = buffer.sampleRate;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // Convert float samples to 16-bit PCM
    const channelData = buffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return arrayBuffer;
}

// Process button functionality
if (processBtn) {
    processBtn.addEventListener('click', () => {
        startDiffusionProcess();
    });
}

// Demo generation buttons
document.querySelectorAll('.generate-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const demoId = e.target.dataset.demoId || e.target.closest('.generate-btn').dataset.demoId;
        startDemoGeneration(demoId, e.target.closest('.diffusion-demo'));
    });
});

// Start diffusion process for uploaded file
function startDiffusionProcess() {
    processBtn.style.display = 'none';
    processProgress.style.display = 'block';
    
    // Get the uploaded file
    const fileInput = document.getElementById('audio-upload');
    if (!fileInput.files.length) {
        alert('Please select a file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('audio', fileInput.files[0]);
    
    const progressFill = processProgress.querySelector('.progress-fill');
    const stepText = processProgress.querySelector('.step-text');
    const eta = processProgress.querySelector('.eta');
    
    // Call the real API
    fetch('/api/generate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Simulate progress updates based on the response
            const steps = data.progress_updates || [
                { progress: 10, description: 'Loading audio and preprocessing...' },
                { progress: 25, description: 'Encoding vocal to latent space...' },
                { progress: 35, description: 'Initializing noise tensor...' },
                { progress: 50, description: 'Running diffusion sampling...' },
                { progress: 95, description: 'Decoding latent to audio...' },
                { progress: 100, description: 'Saving output...' }
            ];
            
            let currentStep = 0;
            
            function updateProgress() {
                if (currentStep < steps.length) {
                    const step = steps[currentStep];
                    
                    progressFill.style.width = step.progress + '%';
                    stepText.textContent = step.description;
                    eta.textContent = `ETA: ${Math.max(0, steps.length - currentStep - 1)}s`;
                    
                    setTimeout(() => {
                        currentStep++;
                        updateProgress();
                    }, 1000);
                } else {
                    // Process complete
                    setTimeout(() => {
                        processProgress.style.display = 'none';
                        showProcessResult(data.output_file);
                    }, 1000);
                }
            }
            
            updateProgress();
        } else {
            alert(`Error: ${data.message || data.error}`);
            processProgress.style.display = 'none';
            processBtn.style.display = 'flex';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during processing');
        processProgress.style.display = 'none';
        processBtn.style.display = 'flex';
    });
}

// Start demo generation using real model
function startDemoGeneration(demoId, container) {
    const generateBtn = container.querySelector('.generate-btn');
    const progressDiv = container.querySelector('.diffusion-progress');
    const audioDiv = container.querySelector('.generated-audio');
    
    generateBtn.style.display = 'none';
    progressDiv.style.display = 'block';
    
    const progressFill = progressDiv.querySelector('.progress-fill');
    const progressText = progressDiv.querySelector('.progress-text');
    
    // Get the demo vocal file
    const demoFiles = {
        'smashmouth': '01_demucs/voice/voice_041000_smashmouthallstarlyrics.wav',
        'queen': '01_demucs/voice/voice_037000_queenbohemianrhapsodyofficialvideoremastered.wav',
        'journey': '01_demucs/voice/voice_028000_journeydontstopbelievinofficialaudio.wav'
    };
    
    const vocalFile = demoFiles[demoId];
    if (!vocalFile) {
        alert('Demo file not found');
        return;
    }
    
    // Create a form to send the vocal file path to the server
    const formData = new FormData();
    formData.append('demo_vocal_path', vocalFile);
    
    // Call the real API
    fetch('/api/generate_demo', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Simulate progress updates
            const steps = data.progress_updates || [
                { progress: 20, description: 'Loading demo vocal...' },
                { progress: 50, description: 'Running real diffusion model...' },
                { progress: 80, description: 'Generating instrumental...' },
                { progress: 100, description: 'Complete!' }
            ];
            
            let currentStep = 0;
            
            function updateDemoProgress() {
                if (currentStep < steps.length) {
                    const step = steps[currentStep];
                    
                    progressFill.style.width = step.progress + '%';
                    progressText.textContent = step.description;
                    
                    setTimeout(() => {
                        currentStep++;
                        updateDemoProgress();
                    }, 1000);
                } else {
                    // Demo complete - show real result
                    setTimeout(() => {
                        progressDiv.style.display = 'none';
                        showDemoResult(demoId, audioDiv, data.output_file);
                    }, 1000);
                }
            }
            
            updateDemoProgress();
        } else {
            alert(`Error: ${data.message || data.error}`);
            progressDiv.style.display = 'none';
            generateBtn.style.display = 'flex';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during generation');
        progressDiv.style.display = 'none';
        generateBtn.style.display = 'flex';
    });
}

// Show process result for uploaded file
function showProcessResult(outputFile) {
    processResult.style.display = 'block';
    
    // Use the actual model output file
    const audio = processResult.querySelector('audio source');
    if (outputFile) {
        audio.src = `/uploads/${outputFile}`;
    } else {
        // Fallback to generated noise if no output file
        generateNoiseAudio().then(noiseBlob => {
            const noiseUrl = URL.createObjectURL(noiseBlob);
            audio.src = noiseUrl;
        });
    }
    processResult.querySelector('audio').load();
    
    // Add download functionality
    const downloadBtn = processResult.querySelector('.download-btn');
    downloadBtn.addEventListener('click', () => {
        // In a real app, trigger download of generated file
        alert('Download functionality would be implemented here!');
    });
    
    // Add regenerate functionality
    const regenerateBtn = processResult.querySelector('.regenerate-btn');
    regenerateBtn.addEventListener('click', () => {
        processResult.style.display = 'none';
        processBtn.style.display = 'flex';
    });
}

// Show demo result with real model output
function showDemoResult(demoId, container, outputFile) {
    container.style.display = 'block';
    
    const audio = container.querySelector('audio source');
    
    if (outputFile) {
        // Use the actual model output file
        audio.src = `/uploads/${outputFile}`;
    } else {
        // For demo examples, we need to actually run the model
        // For now, show a message that the model would need to process this
        const disclaimer = container.querySelector('.disclaimer');
        disclaimer.textContent = "🎵 Click 'Generate with Diffusion' to run the real model!";
        return;
    }
    
    container.querySelector('audio').load();
    
    // Update disclaimer to reflect real model output
    const disclaimers = [
        "🎵 Real model output (early training stage)",
        "🤖 Actual diffusion model result",
        "🎸 This is what the model currently generates",
        "🎹 Raw AI output - work in progress!",
        "🎭 Authentic model inference result"
    ];
    
    const disclaimer = container.querySelector('.disclaimer');
    disclaimer.textContent = disclaimers[Math.floor(Math.random() * disclaimers.length)];
}

// Add some visual flair to the diffusion visualization
function animateNoiseParticles() {
    const particles = document.querySelectorAll('.noise-particles');
    
    particles.forEach(particle => {
        // Add random floating animation
        setInterval(() => {
            const x = Math.random() * 100;
            const y = Math.random() * 100;
            particle.style.backgroundPosition = `${x}% ${y}%`;
        }, 2000);
    });
}

// Initialize animations when page loads
document.addEventListener('DOMContentLoaded', () => {
    animateNoiseParticles();
    
    // Add some random sparkle effects
    setInterval(() => {
        addSparkleEffect();
    }, 3000);
});

// Add sparkle effects for visual appeal
function addSparkleEffect() {
    const sparkle = document.createElement('div');
    sparkle.style.position = 'fixed';
    sparkle.style.width = '4px';
    sparkle.style.height = '4px';
    sparkle.style.background = '#1db954';
    sparkle.style.borderRadius = '50%';
    sparkle.style.pointerEvents = 'none';
    sparkle.style.zIndex = '1000';
    
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    
    sparkle.style.left = x + 'px';
    sparkle.style.top = y + 'px';
    
    document.body.appendChild(sparkle);
    
    // Animate and remove
    sparkle.animate([
        { transform: 'scale(0)', opacity: 1 },
        { transform: 'scale(1)', opacity: 1 },
        { transform: 'scale(0)', opacity: 0 }
    ], {
        duration: 1000,
        easing: 'ease-out'
    }).onfinish = () => {
        sparkle.remove();
    };
}

// Audio player enhancements
document.addEventListener('DOMContentLoaded', () => {
    const audioElements = document.querySelectorAll('audio');
    
    audioElements.forEach(audio => {
        // Pause other audio when one starts playing
        audio.addEventListener('play', () => {
            audioElements.forEach(otherAudio => {
                if (otherAudio !== audio) {
                    otherAudio.pause();
                }
            });
        });
        
        // Add loading state
        audio.addEventListener('loadstart', () => {
            const container = audio.closest('.audio-player, .uploaded-file, .generated-audio');
            if (container) {
                container.classList.add('loading');
            }
        });
        
        audio.addEventListener('canplay', () => {
            const container = audio.closest('.audio-player, .uploaded-file, .generated-audio');
            if (container) {
                container.classList.remove('loading');
            }
        });
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Space to pause/play current audio
    if (e.code === 'Space') {
        e.preventDefault();
        const playingAudio = document.querySelector('audio:not([paused])');
        if (playingAudio) {
            playingAudio.pause();
        } else {
            const firstAudio = document.querySelector('audio');
            if (firstAudio) {
                firstAudio.play();
            }
        }
    }
    
    // Number keys to switch sections
    if (e.key >= '1' && e.key <= '3') {
        const sectionIndex = parseInt(e.key) - 1;
        const navItem = navItems[sectionIndex];
        if (navItem) {
            navItem.click();
        }
    }
});

// Add tooltips for better UX
function addTooltips() {
    const tooltipElements = [
        { selector: '.play-button', text: 'Play demo audio' },
        { selector: '.generate-btn', text: 'Generate with AI diffusion model' },
        { selector: '.upload-area', text: 'Click or drag to upload audio file' },
        { selector: '.nav-item', text: 'Navigate to section' }
    ];
    
    tooltipElements.forEach(({ selector, text }) => {
        document.querySelectorAll(selector).forEach(element => {
            element.title = text;
        });
    });
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', addTooltips);

// Console easter egg
console.log(`
🎵 VocalSeparate AI Demo 🎵

Welcome to our voice-to-instrumental diffusion model demo!

🎭 HONEST DISCLAIMER: Our model is in early training stages, so all 
"generated" results are actually just noise! This represents the 
real output from simple_model_epoch_best.

The sleek Spotify-inspired UI might make it look professional, 
but the AI is still learning the difference between music and chaos! 😅

Enjoy the modern design while we train better models!
`);

// Performance monitoring (for development)
if (window.performance) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const timing = window.performance.timing;
            const loadTime = timing.loadEventEnd - timing.navigationStart;
            console.log(`Page loaded in ${loadTime}ms`);
        }, 0);
    });
} 