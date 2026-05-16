/**
 * main.js — Site enhancements (mobile nav, smooth scroll)
 */
(function() {
  'use strict';

  // Mobile menu toggle
  const menuToggle = document.getElementById('menuToggle');
  const siteNav = document.getElementById('siteNav');

  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', () => {
      siteNav.classList.toggle('open');
    });

    // Close nav on link click
    siteNav.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        siteNav.classList.remove('open');
      });
    });
  }

  // Year auto-update in footer
  const footer = document.querySelector('.site-footer');
  if (footer) {
    const yearSpan = footer.querySelector('.year');
    if (yearSpan) {
      yearSpan.textContent = new Date().getFullYear();
    }
  }
})();
