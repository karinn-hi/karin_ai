
function addDownloadButtonToImage(image) {
    const downloadButton = document.createElement('a');
    downloadButton.innerText = '⬇️';
    downloadButton.href = image.src;
    downloadButton.download = `karinn-ai-${new Date().toISOString()}.png`;
    downloadButton.style.position = 'absolute';
    downloadButton.style.top = '10px';
    downloadButton.style.right = '10px';
    downloadButton.style.backgroundColor = 'rgba(255, 153, 204, 0.8)';
    downloadButton.style.color = 'white';
    downloadButton.style.padding = '5px 10px';
    downloadButton.style.borderRadius = '15px';
    downloadButton.style.textDecoration = 'none';
    downloadButton.style.fontSize = '20px';
    downloadButton.style.zIndex = '100';

    const parent = image.parentElement;
    if (parent && parent.style.position !== 'relative') {
        parent.style.position = 'relative';
    }
    parent.appendChild(downloadButton);
}

function observeGallery() {
    const gallery = document.getElementById('final_gallery');
    if (!gallery) return;

    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'IMG') {
                        addDownloadButtonToImage(node);
                    } else if (node.querySelectorAll) {
                        node.querySelectorAll('img').forEach(addDownloadButtonToImage);
                    }
                });
            }
        });
    });

    observer.observe(gallery, { childList: true, subtree: true });

    // Also add to existing images
    gallery.querySelectorAll('img').forEach(addDownloadButtonToImage);
}

// Run after the UI is built
document.addEventListener('DOMContentLoaded', () => {
    const observer = new MutationObserver((mutations, obs) => {
        const gallery = document.getElementById('final_gallery');
        if (gallery) {
            observeGallery();
            obs.disconnect(); // We found the gallery, no need to observe further
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
