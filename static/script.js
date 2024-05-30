document.addEventListener("DOMContentLoaded", function() {
    const splashScreen = document.getElementById("splash-screen");
    const splashText = document.getElementById("splash-text");
    const mainContent = document.getElementById("main-content");

    var splashAnimation = lottie.loadAnimation({
        container: document.getElementById('splash-animation-container'),
        renderer: 'svg',
        loop: false,
        autoplay: true,
        path: "{{ url_for('static', filename='splash.json') }}"
    });

    splashAnimation.addEventListener('complete', function() {
        splashText.style.opacity = 1; // Show the text after the animation completes

        setTimeout(() => {
            splashScreen.style.opacity = 0; // Fade out splash screen

            setTimeout(() => {
                splashScreen.style.display = 'none';
                mainContent.style.display = 'block'; // Show the main content
            }, 2000); // Allow time for the fade-out effect
        }, 2000); // Keep the text visible for a while
    });
});
