window.addEventListener('scroll', () => {
    const logo = document.getElementById('scroll-logo');
    const rotation = (window.scrollY / 3) % 360;
    logo.style.transform = `rotate(${rotation}deg)`;
});