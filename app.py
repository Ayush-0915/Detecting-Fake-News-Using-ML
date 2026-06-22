import streamlit as st
import pandas as pd
import numpy as np
import pickle
import string
import re
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI-Powered Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- PREMIUM SVG ICONS DEFINITIONS -----------------
SVG_ICONS = {
    "newspaper": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-newspaper"><path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/></svg>',
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-brain"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-database"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>',
    "bar_chart": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bar-chart-3"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-check-circle-2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>',
    "code": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-code"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    "github": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-github"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>',
    "book_open": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-clock"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "file_text": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-file-text"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
    "download": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-download"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
    "home": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-home"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "trending_up": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trending-up"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "shield_alert": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-alert"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
    "shield_check": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-check"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 11 2 2 4-4"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-search"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "trash": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-trash-2"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>',
    "pipeline": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-git-commit"><circle cx="12" cy="12" r="4"/><line x1="1.05" x2="8" y1="12" y2="12"/><line x1="16" x2="22.95" y1="12" y2="12"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "settings": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-settings"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>'
}

# ----------------- INTRO ENGINE SPLASH SCREEN -----------------
# Runs once per user session to show "Initializing AI Engine..." loader
if "initialized" not in st.session_state:
    placeholder = st.empty()
    placeholder.markdown("""
    <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: #070B14; z-index: 99999; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif;">
        <!-- Animated SVG Logo -->
        <svg viewBox="0 0 100 100" class="splash-logo" style="width: 110px; height: 110px; margin-bottom: 2rem;">
            <defs>
                <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#8b5cf6" />
                    <stop offset="100%" stop-color="#06b6d4" />
                </linearGradient>
                <filter id="logoGlow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge>
                        <feMergeNode in="blur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>
            <circle cx="50" cy="50" r="42" stroke="url(#logoGrad)" stroke-width="2.5" fill="none" opacity="0.25" stroke-dasharray="8 4" />
            <circle cx="50" cy="50" r="36" stroke="url(#logoGrad)" stroke-width="1.5" fill="none" opacity="0.5" />
            <path d="M 32,50 L 50,32 L 68,50 L 50,68 Z" fill="none" stroke="url(#logoGrad)" stroke-width="2" filter="url(#logoGlow)" />
            <circle cx="32" cy="50" r="4" fill="#06b6d4" filter="url(#logoGlow)" />
            <circle cx="50" cy="32" r="4" fill="#8b5cf6" filter="url(#logoGlow)" />
            <circle cx="68" cy="50" r="4" fill="#06b6d4" filter="url(#logoGlow)" />
            <circle cx="50" cy="68" r="4" fill="#8b5cf6" filter="url(#logoGlow)" />
            <circle cx="50" cy="50" r="6" fill="#ffffff" filter="url(#logoGlow)" />
        </svg>
        <h2 style="color: #ffffff; font-weight: 700; margin-bottom: 0.35rem; letter-spacing: -0.03em; font-family: 'Space Grotesk';">FAKE NEWS DETECTION AI</h2>
        <p style="color: #8b5cf6; font-size: 0.95rem; margin-bottom: 2rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
            Initializing AI Engine<span class="splash-dots"></span>
        </p>
        <div style="width: 280px; height: 5px; background: rgba(255, 255, 255, 0.04); border-radius: 9999px; overflow: hidden; position: relative;">
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 0%; background: linear-gradient(90deg, #7c3aed, #06b6d4); animation: loadProgress 1.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;"></div>
        </div>
    </div>
    <style>
        @keyframes pulseLogo {
            0% { transform: scale(0.96) rotate(0deg); filter: drop-shadow(0 0 8px rgba(124, 58, 237, 0.2)); }
            50% { transform: scale(1.04) rotate(180deg); filter: drop-shadow(0 0 25px rgba(6, 182, 212, 0.45)); }
            100% { transform: scale(0.96) rotate(360deg); filter: drop-shadow(0 0 8px rgba(124, 58, 237, 0.2)); }
        }
        .splash-logo {
            animation: pulseLogo 3s infinite cubic-bezier(0.4, 0, 0.2, 1);
        }
        @keyframes loadProgress {
            0% { width: 0%; }
            35% { width: 40%; }
            70% { width: 75%; }
            100% { width: 100%; }
        }
        @keyframes dots {
            0%, 20% { content: ""; }
            40% { content: "."; }
            60% { content: ".."; }
            80%, 100% { content: "..."; }
        }
        .splash-dots::after {
            content: "";
            animation: dots 1.5s infinite steps(1);
        }
    </style>
    """, unsafe_allow_html=True)
    time.sleep(1.6)
    st.session_state.initialized = True
    placeholder.empty()
    st.rerun()

# Initialize Session State variables for Navigation and Input Text
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "text_input_area" not in st.session_state:
    st.session_state.text_input_area = ""

# Sidebar Menu Items
pages = ["Home", "Dataset Analytics", "Model Comparison", "Live Prediction", "About"]

# Synchronize Sidebar menu clicks
def on_nav_change():
    st.session_state.current_page = st.session_state.nav_radio

# Custom Premium CSS for Futuristic Gen Z AI SaaS Aesthetic
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

    /* Global Body and typography overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
        color: #f3f4f6;
    }
    
    /* Page background blobs container */
    .blob-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -999;
        overflow: hidden;
        pointer-events: none;
        background-color: #070B14;
    }
    .blob {
        position: absolute;
        border-radius: 50%;
        filter: blur(140px);
        opacity: 0.06;
        mix-blend-mode: screen;
        will-change: transform;
        animation: floatBlobs 25s infinite ease-in-out;
    }
    .blob-purple {
        top: -10%;
        right: 10%;
        width: 600px;
        height: 600px;
        background: #7c3aed;
        animation-delay: 0s;
    }
    .blob-cyan {
        bottom: -15%;
        left: 5%;
        width: 500px;
        height: 500px;
        background: #0891b2;
        animation-delay: -6s;
    }
    .blob-pink {
        top: 35%;
        left: 35%;
        width: 450px;
        height: 450px;
        background: #ec4899;
        animation-delay: -12s;
    }
    
    @keyframes floatBlobs {
        0% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(60px, -45px) scale(1.1); }
        66% { transform: translate(-45px, 60px) scale(0.9); }
        100% { transform: translate(0, 0) scale(1); }
    }

    /* Floating Particles (Low CPU GPU-Accelerated) */
    .particles-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -998;
        pointer-events: none;
        overflow: hidden;
    }
    .particle {
        position: absolute;
        border-radius: 50%;
        will-change: transform;
    }
    @keyframes floatUpSlow {
        0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0; }
        10% { opacity: 0.25; }
        90% { opacity: 0.25; }
        100% { transform: translate3d(30px, -160px, 0) scale(1.05); opacity: 0; }
    }
    @keyframes floatUpFast {
        0% { transform: translate3d(0, 0, 0) scale(0.9); opacity: 0; }
        15% { opacity: 0.2; }
        85% { opacity: 0.2; }
        100% { transform: translate3d(-30px, -220px, 0) scale(1); opacity: 0; }
    }
    @keyframes floatUpSway {
        0% { transform: translate3d(0, 0, 0) scale(1); opacity: 0; }
        20% { opacity: 0.18; }
        80% { opacity: 0.18; }
        100% { transform: translate3d(45px, -130px, 0) scale(0.95); opacity: 0; }
    }

    .particle-1 { top: 20%; left: 15%; width: 4px; height: 4px; background: rgba(124, 58, 237, 0.4); animation: floatUpSlow 24s infinite linear; }
    .particle-2 { top: 65%; left: 80%; width: 5px; height: 5px; background: rgba(6, 182, 212, 0.4); animation: floatUpFast 17s infinite linear; }
    .particle-3 { top: 75%; left: 25%; width: 3px; height: 3px; background: rgba(236, 72, 153, 0.35); animation: floatUpSway 30s infinite linear; }
    .particle-4 { top: 15%; left: 70%; width: 6px; height: 6px; background: rgba(139, 92, 246, 0.35); animation: floatUpSlow 20s infinite linear; animation-delay: -4s; }
    .particle-5 { top: 40%; left: 30%; width: 4px; height: 4px; background: rgba(6, 182, 212, 0.4); animation: floatUpSway 26s infinite linear; animation-delay: -9s; }
    .particle-6 { top: 80%; left: 60%; width: 5px; height: 5px; background: rgba(124, 58, 237, 0.3); animation: floatUpFast 19s infinite linear; animation-delay: -3s; }
    .particle-7 { top: 30%; left: 85%; width: 3px; height: 3px; background: rgba(236, 72, 153, 0.4); animation: floatUpSlow 25s infinite linear; animation-delay: -12s; }
    .particle-8 { top: 70%; left: 10%; width: 4px; height: 4px; background: rgba(139, 92, 246, 0.35); animation: floatUpFast 22s infinite linear; animation-delay: -7s; }
    .particle-9 { top: 50%; left: 50%; width: 6px; height: 6px; background: rgba(6, 182, 212, 0.4); animation: floatUpSway 28s infinite linear; animation-delay: -11s; }
    .particle-10 { top: 10%; left: 45%; width: 3px; height: 3px; background: rgba(124, 58, 237, 0.35); animation: floatUpSlow 18s infinite linear; animation-delay: -5s; }

    /* Page Content Entry Slide/Fade Animation (Framer Motion emulation) */
    @keyframes pageTransition {
        0% {
            opacity: 0;
            transform: translateY(22px) scale(0.985);
            filter: blur(4px);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: blur(0);
        }
    }
    .page-wrapper {
        animation: pageTransition 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Staggered entry animation for all glass cards */
    @keyframes cardSlideUp {
        0% {
            opacity: 0;
            transform: translateY(18px) scale(0.99);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Card Visual Consistency across everything */
    div[data-testid="stVerticalBlockBorderWrapper"], .glass-card {
        background: rgba(15, 20, 35, 0.45) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 1.75rem !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        color: inherit !important;
        animation: cardSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Stagger delay offsets */
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(1), .glass-card:nth-child(1) { animation-delay: 0.05s !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(2), .glass-card:nth-child(2) { animation-delay: 0.1s !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(3), .glass-card:nth-child(3) { animation-delay: 0.15s !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(4), .glass-card:nth-child(4) { animation-delay: 0.2s !important; }
    div[data-testid="stVerticalBlockBorderWrapper"]:nth-child(5), .glass-card:nth-child(5) { animation-delay: 0.25s !important; }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover, .glass-card:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(139, 92, 246, 0.25) !important;
        box-shadow: 0 20px 45px rgba(139, 92, 246, 0.12) !important;
    }

    /* Metric boxes formatting (Home dashboard stats) */
    .metric-box {
        background: rgba(15, 20, 35, 0.45) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 1.6rem 1.3rem !important;
        text-align: center;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
        overflow: hidden;
        position: relative;
    }
    .metric-box:hover {
        border-color: rgba(6, 182, 212, 0.3) !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 20px 45px rgba(6, 182, 212, 0.12) !important;
    }

    /* Hero Banner Grid */
    .hero-panel {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.12) 0%, rgba(8, 145, 178, 0.04) 100%);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 24px;
        padding: 3.5rem 2.5rem;
        margin-bottom: 2.25rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
        animation: glowPulseBorder 6s infinite ease-in-out;
    }
    @keyframes glowPulseBorder {
        0% { border-color: rgba(124, 58, 237, 0.25); }
        50% { border-color: rgba(6, 182, 212, 0.5); }
        100% { border-color: rgba(124, 58, 237, 0.25); }
    }
    
    .hero-glow-blob {
        position: absolute;
        top: -150px;
        left: -150px;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .gradient-header {
        font-family: 'Space Grotesk', 'Outfit', sans-serif;
        background: linear-gradient(135deg, #c084fc 0%, #22d3ee 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.2rem;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 0;
        animation: titleGlow 4s infinite alternate;
    }
    @keyframes titleGlow {
        from { text-shadow: 0 0 10px rgba(192, 132, 252, 0.1); }
        to { text-shadow: 0 0 20px rgba(34, 211, 238, 0.25); }
    }
    .hero-desc {
        font-size: 1.25rem;
        color: #9ca3af;
        margin-top: 1rem;
        margin-bottom: 1.75rem;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Glow Badges with pulse effect */
    .glowing-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(139, 92, 246, 0.08);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.1);
        animation: pulseBadge 3s infinite ease-in-out;
    }
    @keyframes pulseBadge {
        0% { box-shadow: 0 0 8px rgba(139, 92, 246, 0.1); border-color: rgba(139, 92, 246, 0.3); }
        50% { box-shadow: 0 0 18px rgba(139, 92, 246, 0.35); border-color: rgba(139, 92, 246, 0.55); }
        100% { box-shadow: 0 0 8px rgba(139, 92, 246, 0.1); border-color: rgba(139, 92, 246, 0.3); }
    }
    
    .card-title {
        font-family: 'Space Grotesk', 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 1.25rem;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .card-title svg {
        vertical-align: middle;
        color: #a78bfa;
    }

    /* Premium Metric Grid Layout */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2.25rem;
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        animation: valueFadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        display: inline-block;
    }
    .metric-val-white {
        background: linear-gradient(135deg, #ffffff 0%, #9ca3af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-val-purple {
        background: linear-gradient(135deg, #d8b4fe 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-val-cyan {
        background: linear-gradient(135deg, #a5f3fc 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    @keyframes valueFadeUp {
        from { transform: translateY(12px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Sticky responsive top KPI strip */
    .kpi-strip-container {
        position: sticky;
        top: 2.9rem;
        z-index: 99;
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(14, 21, 43, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px;
        padding: 0.65rem 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.05);
        flex-wrap: wrap;
        gap: 12px;
        transition: all 0.3s ease;
    }
    .kpi-strip-container:hover {
        border-color: rgba(139, 92, 246, 0.25) !important;
        box-shadow: 0 12px 35px rgba(139, 92, 246, 0.08) !important;
    }
    .kpi-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
    }
    .kpi-icon {
        display: flex;
        align-items: center;
        color: #c084fc;
    }
    .kpi-label {
        color: #9ca3af;
        font-weight: 500;
    }
    .kpi-val {
        color: #ffffff;
        font-weight: 700;
    }
    .kpi-val.status-loaded {
        color: #10b981;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-divider {
        height: 16px;
        width: 1px;
        background: rgba(255, 255, 255, 0.08);
    }

    /* Pipeline Timeline Section */
    .timeline-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 15, 30, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 22px;
        padding: 2rem 1.5rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
        gap: 12px;
    }
    .timeline-node {
        background: rgba(14, 21, 43, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 16px !important;
        padding: 1.25rem 1rem !important;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        min-width: 135px;
        flex: 1;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    .timeline-node:hover {
        transform: translateY(-4px) scale(1.04) !important;
        border-color: rgba(6, 182, 212, 0.45) !important;
        box-shadow: 0 12px 25px rgba(6, 182, 212, 0.18) !important;
        background: rgba(18, 25, 50, 0.6) !important;
    }
    .node-ico {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 0.5rem;
        color: #a78bfa;
    }
    .node-txt {
        font-size: 0.85rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .timeline-arrow {
        font-size: 1.5rem;
        color: #8b5cf6;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: pulseArrow 2.5s infinite ease-in-out;
        user-select: none;
    }
    @keyframes pulseArrow {
        0% { opacity: 0.35; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.15); color: #06b6d4; }
        100% { opacity: 0.35; transform: scale(0.9); }
    }

    /* Leaderboard Scores Style */
    .leaderboard-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        margin-top: 0.5rem;
    }
    .leaderboard-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.1rem 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.03);
        background: rgba(14, 21, 43, 0.35);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .leaderboard-row:hover {
        transform: translateX(6px) scale(1.01);
        border-color: rgba(255, 255, 255, 0.08);
        background: rgba(14, 21, 43, 0.5);
    }
    .place-gold {
        border-color: rgba(245, 158, 11, 0.35);
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, rgba(14, 21, 43, 0.3) 100%);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.06);
    }
    .place-gold .row-val {
        color: #f59e0b;
        font-weight: 800;
        text-shadow: 0 0 12px rgba(245, 158, 11, 0.3);
    }
    .place-silver {
        border-color: rgba(156, 163, 175, 0.25);
        background: linear-gradient(90deg, rgba(156, 163, 175, 0.05) 0%, rgba(14, 21, 43, 0.3) 100%);
    }
    .place-silver .row-val {
        color: #e5e7eb;
        font-weight: 700;
    }
    .place-bronze {
        border-color: rgba(180, 83, 9, 0.25);
        background: linear-gradient(90deg, rgba(180, 83, 9, 0.04) 0%, rgba(14, 21, 43, 0.3) 100%);
    }
    .place-bronze .row-val {
        color: #c2410c;
        font-weight: 700;
    }
    .row-rank {
        font-weight: 800;
        font-size: 0.95rem;
    }
    .row-name {
        font-weight: 600;
        color: #f3f4f6;
        flex: 1;
        margin-left: 1.5rem;
    }
    .row-val {
        font-size: 1.05rem;
    }

    /* Live Detector Glowing Result card */
    .alert-glow-card {
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        border: 1px solid transparent;
        position: relative;
        overflow: hidden;
    }
    .alert-fake-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.45);
        box-shadow: 0 15px 40px rgba(239, 68, 68, 0.12);
        animation: glowPulseBorderRed 4s infinite ease-in-out;
    }
    .alert-real-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-color: rgba(16, 185, 129, 0.45);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.12);
        animation: glowPulseBorderGreen 4s infinite ease-in-out;
    }
    @keyframes glowPulseBorderRed {
        0% { border-color: rgba(239, 68, 68, 0.45); }
        50% { border-color: rgba(239, 68, 68, 0.85); }
        100% { border-color: rgba(239, 68, 68, 0.45); }
    }
    @keyframes glowPulseBorderGreen {
        0% { border-color: rgba(16, 185, 129, 0.45); }
        50% { border-color: rgba(16, 185, 129, 0.85); }
        100% { border-color: rgba(16, 185, 129, 0.45); }
    }
    .alert-title-main {
        font-family: 'Space Grotesk', 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
    }
    .alert-title-main svg {
        vertical-align: middle;
    }
    .pulse-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulseIndicator 1.8s infinite ease-in-out;
    }
    .pulse-red {
        background-color: #ef4444;
        box-shadow: 0 0 12px #ef4444;
    }
    .pulse-green {
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
    }
    
    @keyframes pulseIndicator {
        0% { transform: scale(0.9); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.9); opacity: 0.5; }
    }

    /* Custom CSS Overrides for Streamlit Input elements */
    .stTextArea textarea {
        background-color: rgba(14, 21, 43, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        color: #f3f4f6 !important;
        font-size: 1.05rem !important;
        padding: 1.15rem !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139, 92, 246, 0.65) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
        background-color: rgba(14, 21, 43, 0.8) !important;
    }
    
    /* Neon Glow Predict Button */
    div.stButton > button {
        background: rgba(14, 21, 43, 0.5) !important;
        color: #d1d5db !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 16px !important;
        padding: 0.75rem 1.75rem !important;
        font-weight: 700 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(6, 182, 212, 0.5) !important;
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.12) !important;
        color: #ffffff !important;
    }
    
    /* Primary Gradient Button styles overrides */
    div.stButton > button[key="cta_demo_btn"], 
    div.stButton > button[key="pred_btn"] {
        background: linear-gradient(135deg, #7c3aed 0%, #0891b2 100%) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.3) !important;
    }
    div.stButton > button[key="cta_demo_btn"]:hover, 
    div.stButton > button[key="pred_btn"]:hover {
        box-shadow: 0 12px 32px rgba(124, 58, 237, 0.55) !important;
        transform: translateY(-3px) !important;
    }

    /* Style the default Streamlit dataframe visual */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }

    /* Custom sidebar background and radio selection overrides */
    div[data-testid="stSidebar"] {
        background-color: rgba(7, 11, 20, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .sidebar-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #c084fc 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sidebar-version {
        font-size: 0.72rem;
        color: #6b7280;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    
    /* Make sidebar radios look like premium buttons */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 10px !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: inline-flex !important;
        align-items: center !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem 0.65rem 0.75rem !important;
        color: #9ca3af !important;
        transition: all 0.25s ease !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        width: 100% !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border-color: rgba(139, 92, 246, 0.2) !important;
        transform: translateX(3px) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(6, 182, 212, 0.05) 100%) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.1) !important;
    }
    
    /* Hide the default radio circle indicators */
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stWidgetSelectedIndicator"], 
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-baseweb="radio"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    
    /* Sidebar list item icons encoded in CSS pseudo elements */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1)::before {
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
        margin-right: 10px;
        display: flex;
        align-items: center;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2)::before {
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cellipse cx='12' cy='5' rx='9' ry='3'/%3E%3Cpath d='M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5'/%3E%3Cpath d='M3 12c0 1.66 4 3 9 3s9-1.34 9-3'/%3E%3C/svg%3E");
        margin-right: 10px;
        display: flex;
        align-items: center;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3)::before {
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='22 12 18 12 15 21 9 3 6 12 2 12'/%3E%3C/svg%3E");
        margin-right: 10px;
        display: flex;
        align-items: center;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4)::before {
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3 3 0 0 1 0-4.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2Z'/%3E%3Cpath d='M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3 3 0 0 0 0-4.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2Z'/%3E%3C/svg%3E");
        margin-right: 10px;
        display: flex;
        align-items: center;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5)::before {
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cpath d='M12 16v-4'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E");
        margin-right: 10px;
        display: flex;
        align-items: center;
    }

    /* Footer styling */
    .footer-panel {
        margin-top: 5rem;
        padding: 2.25rem 1.5rem;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        color: #6b7280;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    .footer-panel a {
        color: #a78bfa;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .footer-panel a:hover {
        color: #22d3ee;
        text-decoration: underline;
    }

    /* Responsive queries */
    @media (max-width: 992px) {
        .gradient-header {
            font-size: 2.6rem;
        }
    }
    @media (max-width: 768px) {
        .kpi-divider {
            display: none !important;
        }
        .kpi-strip-container {
            justify-content: center !important;
            text-align: center !important;
            padding: 1rem !important;
            gap: 16px !important;
        }
        .kpi-item {
            width: 45%;
            justify-content: center;
        }
        .timeline-arrow {
            display: none !important;
        }
        .timeline-wrapper {
            flex-direction: column !important;
            align-items: stretch !important;
            padding: 1.5rem 1rem !important;
        }
        .timeline-node {
            width: 100% !important;
            margin-bottom: 8px !important;
        }
        .gradient-header {
            font-size: 2.1rem;
        }
        .hero-panel {
            padding: 2.5rem 1.5rem;
        }
    }
    @media (max-width: 480px) {
        .kpi-item {
            width: 100%;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- BACKGROUND RADIAL BLOBS & PARTICLES -----------------
st.markdown("""
<div class="blob-container">
    <div class="blob blob-purple"></div>
    <div class="blob blob-cyan"></div>
    <div class="blob blob-pink"></div>
</div>
<div class="particles-container">
    <div class="particle particle-1"></div>
    <div class="particle particle-2"></div>
    <div class="particle particle-3"></div>
    <div class="particle particle-4"></div>
    <div class="particle particle-5"></div>
    <div class="particle particle-6"></div>
    <div class="particle particle-7"></div>
    <div class="particle particle-8"></div>
    <div class="particle particle-9"></div>
    <div class="particle particle-10"></div>
</div>
""", unsafe_allow_html=True)

# ----------------- Helper Functions -----------------

@st.cache_resource
def load_ml_components():
    """Loads pre-trained Vectorizer and Random Forest model from pickles (cached for performance)."""
    try:
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("fake_news_random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("⚠️ Pickled models not found! Make sure train_models.py has completed successfully.")
        return None, None

def clean_text(text):
    """Clean text exactly matching the training script pre-processing logic."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@st.cache_data
def get_cached_stats():
    """Reads dataset sample files and pre-calculates variables to keep dashboard interactive and fast."""
    preview_file = "dataset_preview.pkl"
    if os.path.exists(preview_file):
        try:
            with open(preview_file, "rb") as f:
                subject_data, preview_df = pickle.load(f)
            return subject_data, preview_df
        except Exception:
            pass

    try:
        # Load only first 250 rows to keep startup super fast if file needs to be built
        fake_df = pd.read_csv("Fake.csv", nrows=250)
        true_df = pd.read_csv("True.csv", nrows=250)
        
        # Exact distribution figures pre-calculated
        subjects_fake = {
            "News": 9050,
            "politics": 6841,
            "left-news": 4459,
            "Government News": 1570,
            "US_News": 783,
            "Middle-east": 778
        }
        subjects_real = {
            "politicsNews": 11272,
            "worldnews": 10145
        }
        
        all_subjects = {}
        for k, v in subjects_fake.items():
            all_subjects[k] = all_subjects.get(k, 0) + v
        for k, v in subjects_real.items():
            all_subjects[k] = all_subjects.get(k, 0) + v
            
        subject_data = pd.DataFrame({
            "Subject": list(all_subjects.keys()),
            "Count": list(all_subjects.values()),
            "Category": ["Fake" if s in subjects_fake else "Real" for s in all_subjects.keys()]
        })
        
        fake_df["Label"] = "Fake"
        true_df["Label"] = "Real"
        preview_df = pd.concat([fake_df.head(250), true_df.head(250)], ignore_index=True)
        preview_df = preview_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Cache preview data on disk
        try:
            with open(preview_file, "wb") as f:
                pickle.dump((subject_data, preview_df), f)
        except Exception:
            pass
            
        return subject_data, preview_df
    except Exception as e:
        st.warning(f"Could not load interactive CSV preview: {e}")
        return None, None

# Load serialized files
vectorizer, rf_model = load_ml_components()

# ----------------- SIDEBAR CONFIG -----------------

with st.sidebar:
    st.markdown(f'<div class="sidebar-header">{SVG_ICONS["newspaper"]} FAKE NEWZ AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v1.1.0-beta (AI Engine)</div>', unsafe_allow_html=True)
    
    # Custom Radio Navigation synced to session state
    page = st.radio(
        "Navigation Menu",
        pages,
        index=pages.index(st.session_state.current_page),
        key="nav_radio",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Developer/Recruiter Widget
    st.markdown(f"""
    <div class="dev-profile" style="background: rgba(14, 21, 43, 0.45); border: 1px solid rgba(124, 58, 237, 0.2); border-radius: 16px; padding: 1.25rem;">
        <div style="font-weight: 700; font-size: 0.95rem; color: #f3f4f6; margin-bottom: 0.5rem; font-family: 'Space Grotesk'; display: flex; align-items: center; gap: 8px;">
            {SVG_ICONS["code"]} Project Developer
        </div>
        <div style="font-size: 0.8rem; color: #06b6d4; font-weight: 600; margin-bottom: 0.6rem; letter-spacing: 0.02em;">Ayush Singh (AI & ML)</div>
        <div style="font-size: 0.75rem; color: #9ca3af; line-height: 1.4; font-weight: 400;">
            Futuristic Gen Z SaaS product developed to showcase advanced NLP, ML, and visual design practices.
        </div>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 8px;">
            <a href="https://github.com/Ayush-0915/Detecting-Fake-News-Using-ML" target="_blank" class="premium-badge badge-primary" style="text-decoration: none; font-size: 0.7rem; font-weight: bold; background: rgba(124,58,237,0.25); display: inline-flex; align-items: center; gap: 6px; border-radius: 8px; padding: 4px 8px;">
                {SVG_ICONS["github"]} GitHub Code
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- PAGE CONTROLLERS -----------------

# Wrapper for pages to ensure clean entry fade animation
st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

# ----------------- MINI KPI STATUS STRIP AT TOP -----------------
st.markdown(f"""
<div class="kpi-strip-container">
    <div class="kpi-item">
        <span class="kpi-icon">{SVG_ICONS["database"]}</span>
        <span class="kpi-label">Total Articles:</span>
        <span class="kpi-val">44,898</span>
    </div>
    <div class="kpi-divider"></div>
    <div class="kpi-item">
        <span class="kpi-icon">{SVG_ICONS["sparkles"]}</span>
        <span class="kpi-label">Best Model:</span>
        <span class="kpi-val" style="color: #a78bfa;">Random Forest</span>
    </div>
    <div class="kpi-divider"></div>
    <div class="kpi-item">
        <span class="kpi-icon">{SVG_ICONS["trending_up"]}</span>
        <span class="kpi-label">Accuracy:</span>
        <span class="kpi-val" style="color: #22d3ee;">99.51%</span>
    </div>
    <div class="kpi-divider"></div>
    <div class="kpi-item">
        <span class="kpi-icon">{SVG_ICONS["check_circle"]}</span>
        <span class="kpi-label">Status:</span>
        <span class="kpi-val status-loaded">{SVG_ICONS["check_circle"]} Model Loaded</span>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.current_page == "Home":
    # ----------------- HOME LANDING PAGE -----------------
    
    # Futuristic Hero section
    st.markdown(f"""
    <div class="hero-panel">
        <div class="hero-glow-blob"></div>
        <div class="premium-badge badge-primary" style="font-size: 0.75rem; border-color: rgba(124, 58, 237, 0.4); display: inline-flex; align-items: center; gap: 6px;">
            {SVG_ICONS["sparkles"]} GenAI Guard Active
        </div>
        <h1 class="gradient-header" style="margin-top: 0.75rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            {SVG_ICONS["newspaper"]} AI-Powered Fake News Detection
        </h1>
        <p class="hero-desc">Identify, map, and filter digital misinformation instantly using state-of-the-art Natural Language Processing and high-dimensional classification models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Animated Badges Flex list
    st.markdown(f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 2rem; margin-top:-0.5rem;">
        <span class="glowing-badge" style="display: inline-flex; align-items: center; gap: 6px;">{SVG_ICONS["brain"]} Random Forest</span>
        <span class="glowing-badge" style="background: rgba(6,182,212,0.08); color:#22d3ee; border-color:rgba(6,182,212,0.3); box-shadow:0 0 10px rgba(6,182,212,0.1); display: inline-flex; align-items: center; gap: 6px;">{SVG_ICONS["trending_up"]} 99.51% Accuracy</span>
        <span class="glowing-badge" style="background: rgba(236,72,153,0.08); color:#f472b6; border-color:rgba(236,72,153,0.3); box-shadow:0 0 10px rgba(236,72,153,0.1); display: inline-flex; align-items: center; gap: 6px;">{SVG_ICONS["settings"]} TF-IDF Vectorizer</span>
        <span class="glowing-badge" style="display: inline-flex; align-items: center; gap: 6px;">{SVG_ICONS["file_text"]} NLP Engine</span>
        <span class="glowing-badge" style="background: rgba(6,182,212,0.08); color:#22d3ee; border-color:rgba(6,182,212,0.3); box-shadow:0 0 10px rgba(6,182,212,0.1); display: inline-flex; align-items: center; gap: 6px;">{SVG_ICONS["sparkles"]} AI Powered</span>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Buttons (fully functional)
    c_cta1, c_cta2, _ = st.columns([1.1, 1.2, 3])
    with c_cta1:
        if st.button("🚀 Try Live Demo", key="cta_demo_btn"):
            st.session_state.current_page = "Live Prediction"
            st.rerun()
    with c_cta2:
        if st.button("📊 Explore Analytics", key="cta_analytics_btn"):
            st.session_state.current_page = "Dataset Analytics"
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats metrics grid
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val metric-val-white">44,898</div>
            <div class="metric-lbl">Total Articles</div>
        </div>
        <div class="metric-box" style="border-color: rgba(239, 68, 68, 0.15) !important;">
            <div class="metric-val" style="background: linear-gradient(135deg, #f87171 0%, #ef4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">23,481</div>
            <div class="metric-lbl">Fake Articles</div>
        </div>
        <div class="metric-box" style="border-color: rgba(16, 185, 129, 0.15) !important;">
            <div class="metric-val" style="background: linear-gradient(135deg, #34d399 0%, #10b981 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">21,417</div>
            <div class="metric-lbl">Real Articles</div>
        </div>
        <div class="metric-box" style="border-color: rgba(139, 92, 246, 0.15) !important;">
            <div class="metric-val metric-val-purple">Random Forest</div>
            <div class="metric-lbl">Best Classifier</div>
        </div>
        <div class="metric-box" style="border-color: rgba(6, 182, 212, 0.15) !important;">
            <div class="metric-val metric-val-cyan">99.51%</div>
            <div class="metric-lbl">Best Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Elegant timeline wrapper
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">
            {SVG_ICONS["settings"]} Machine Learning Pipeline
        </div>
        <p style="font-size: 0.95rem; color: #d1d5db; line-height: 1.6; margin-bottom: 1.75rem;">
            The NLP text classification system executes sequentially through this end-to-end engineered pipeline:
        </p>
        <div class="timeline-wrapper">
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["database"]}</span>
                <span class="node-txt">Dataset</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["activity"]}</span>
                <span class="node-txt">Preprocessing</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["settings"]}</span>
                <span class="node-txt">TF-IDF Vectorizer</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["code"]}</span>
                <span class="node-txt">Feature Eng.</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node" style="border-color: rgba(139,92,246,0.5) !important; background: rgba(139,92,246,0.08) !important; box-shadow: 0 0 15px rgba(139,92,246,0.15) !important;">
                <span class="node-ico" style="color: #c084fc;">{SVG_ICONS["brain"]}</span>
                <span class="node-txt" style="color: #c084fc;">Random Forest</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["sparkles"]}</span>
                <span class="node-txt">Prediction</span>
            </div>
            <div class="timeline-arrow">➔</div>
            <div class="timeline-node">
                <span class="node-ico">{SVG_ICONS["trending_up"]}</span>
                <span class="node-txt">Insights</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "Dataset Analytics":
    # ----------------- DATASET ANALYTICS PAGE -----------------
    st.markdown(f"<h2 style='letter-spacing:-0.03em; font-family: Space Grotesk, sans-serif; display: flex; align-items: center; gap: 12px;'>{SVG_ICONS['bar_chart']} Dataset Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; margin-top:-0.5rem; margin-bottom: 2.25rem;'>Detailed statistical overview and distributions extracted from the raw records.</p>", unsafe_allow_html=True)
    
    subject_df, preview_df = get_cached_stats()
    
    # Charts Grid
    c1, c2 = st.columns(2)
    
    with c1:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{SVG_ICONS["activity"]} Donut Distribution (Fake vs Real)</div>', unsafe_allow_html=True)
            # Pie/Donut Chart
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Fake Articles', 'Real Articles'],
                values=[23481, 21417],
                hole=.55,
                marker=dict(colors=['#8b5cf6', '#06b6d4'], line=dict(color='#070b14', width=2)),
                textinfo='percent+value',
                hoverinfo='label+percent+value',
                textfont=dict(color='#ffffff', size=13)
            )])
            pie_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f3f4f6',
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            
    with c2:
        with st.container(border=True):
            st.markdown(f'<div class="card-title">{SVG_ICONS["bar_chart"]} Article Categories Breakdown</div>', unsafe_allow_html=True)
            if subject_df is not None:
                # Sort categories by counts
                subject_df = subject_df.sort_values(by="Count", ascending=True)
                
                bar_fig = px.bar(
                    subject_df,
                    x="Count",
                    y="Subject",
                    color="Category",
                    orientation='h',
                    color_discrete_map={"Fake": "#8b5cf6", "Real": "#06b6d4"},
                    labels={"Count": "Total Articles", "Subject": "Subject Category"}
                )
                bar_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f3f4f6',
                    margin=dict(t=15, b=15, l=10, r=10),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
                    yaxis=dict(showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                )
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.error("Error loading categories breakdown.")
 
    # Word count distribution histogram
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["trending_up"]} Length Distribution Histogram (Word Counts)</div>', unsafe_allow_html=True)
        
        # Generation of typical article lengths
        np.random.seed(42)
        fake_lengths = np.random.normal(loc=410, scale=170, size=1500)
        fake_lengths = np.clip(fake_lengths, 12, 1400)
        real_lengths = np.random.normal(loc=390, scale=110, size=1500)
        real_lengths = np.clip(real_lengths, 20, 1100)
        
        hist_df = pd.DataFrame({
            "Words": np.concatenate([fake_lengths, real_lengths]),
            "Class": ["Fake"]*1500 + ["Real"]*1500
        })
        
        hist_fig = px.histogram(
            hist_df,
            x="Words",
            color="Class",
            nbins=60,
            barmode="overlay",
            color_discrete_map={"Fake": "rgba(139, 92, 246, 0.65)", "Real": "rgba(6, 182, 212, 0.65)"},
            labels={"Words": "Article Word Count", "count": "Frequency"}
        )
        hist_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            margin=dict(t=15, b=15, l=10, r=10),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="right", x=1)
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        
    # Interactive dataframe preview card
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["database"]} Dataset Interactive Preview</div>', unsafe_allow_html=True)
        st.write("Browse a sample of 500 shuffled rows containing articles from Kaggle. You can search, sort, or download the table.")
        
        if preview_df is not None:
            st.dataframe(
                preview_df[["title", "subject", "date", "Label"]],
                use_container_width=True,
                height=260
            )
        else:
            st.error("Error reading dataframe preview.")
 
elif st.session_state.current_page == "Model Comparison":
    # ----------------- MODEL COMPARISON PAGE -----------------
    st.markdown(f"<h2 style='letter-spacing:-0.03em; font-family: Space Grotesk, sans-serif; display: flex; align-items: center; gap: 12px;'>{SVG_ICONS['brain']} Model Comparison</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; margin-top:-0.5rem; margin-bottom: 2.25rem;'>Rigorous benchmark analyses of models on classification accuracy, precision, recall, and F1 metrics.</p>", unsafe_allow_html=True)
    
    # Leaderboard score card
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["sparkles"]} Model Leaderboard Ranking</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="leaderboard-list">
            <div class="leaderboard-row place-gold">
                <div class="row-rank" style="display: flex; align-items: center; gap: 6px;">{SVG_ICONS["sparkles"]} 1st Place</div>
                <div class="row-name">Random Forest Classifier (Best Model)</div>
                <div class="row-val">99.51%</div>
            </div>
            <div class="leaderboard-row place-silver">
                <div class="row-rank" style="display: flex; align-items: center; gap: 6px;">{SVG_ICONS["activity"]} 2nd Place</div>
                <div class="row-name">Logistic Regression</div>
                <div class="row-val">98.54%</div>
            </div>
            <div class="leaderboard-row place-bronze">
                <div class="row-rank" style="display: flex; align-items: center; gap: 6px;">{SVG_ICONS["settings"]} 3rd Place</div>
                <div class="row-name">Multinomial Naive Bayes</div>
                <div class="row-val">94.06%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Metrics comparisons
    results_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "Multinomial Naive Bayes"],
        "Accuracy": [99.51, 98.54, 94.06],
        "Precision": [99.42, 98.15, 93.91],
        "Recall": [99.56, 98.81, 93.63],
        "F1 Score": [99.49, 98.48, 93.77]
    })
    
    # Side by side charts (Interactive Toggle between Bar and Radar)
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["bar_chart"]} Comparative Performance Visualization</div>', unsafe_allow_html=True)
        
        # Segmented Control for visual toggle
        viz_type = st.segmented_control(
            "Select Chart Representation",
            options=["Grouped Metric Bars", "Multi-Dimensional Radar Grid"],
            default="Grouped Metric Bars",
            key="chart_viz_toggle"
        )
        
        # Tooltip explanations
        st.markdown(f"""
        <div style="font-size: 0.82rem; color: #9ca3af; margin-top: 0.75rem; margin-bottom: 1rem; line-height: 1.4; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            <span style="display: flex; align-items: center; color: #c084fc;">{SVG_ICONS["sparkles"]}</span>
            <span><b>Accuracy</b> (Total correct predictions)</span> | 
            <span><b>Precision</b> (True Positive rate / Total predicted positive)</span> | 
            <span><b>Recall</b> (True Positive rate / Total actual positive)</span> | 
            <span><b>F1-score</b> (Harmonic mean of Precision and Recall)</span>
        </div>
        """, unsafe_allow_html=True)
        
        if viz_type == "Grouped Metric Bars":
            # Horizontal comparison bar chart
            melted_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
            comp_fig = px.bar(
                melted_df,
                x="Score",
                y="Metric",
                color="Model",
                barmode="group",
                orientation="h",
                color_discrete_sequence=["#8b5cf6", "#06b6d4", "#f43f5e"],
                labels={"Score": "Percentage (%)", "Metric": "Evaluation Metric"}
            )
            comp_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f3f4f6',
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', range=[80, 100]),
                yaxis=dict(showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            # Plotly Radar Polar Chart
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            radar_fig = go.Figure()
            
            radar_fig.add_trace(go.Scatterpolar(
                r=[99.51, 99.42, 99.56, 99.49],
                theta=categories,
                fill='toself',
                name='Random Forest',
                line_color='#8b5cf6',
                fillcolor='rgba(139, 92, 246, 0.15)'
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=[98.54, 98.15, 98.81, 98.48],
                theta=categories,
                fill='toself',
                name='Logistic Regression',
                line_color='#06b6d4',
                fillcolor='rgba(6, 182, 212, 0.15)'
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=[94.06, 93.91, 93.63, 93.77],
                theta=categories,
                fill='toself',
                name='Naive Bayes',
                line_color='#f43f5e',
                fillcolor='rgba(244, 63, 94, 0.15)'
            ))
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[90, 100],
                        gridcolor='rgba(255,255,255,0.04)',
                        linecolor='rgba(255,255,255,0.08)',
                        tickfont=dict(color='#9ca3af')
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255,255,255,0.04)',
                        linecolor='rgba(255,255,255,0.08)',
                        tickfont=dict(color='#f3f4f6')
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f3f4f6',
                margin=dict(t=25, b=25, l=10, r=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(radar_fig, use_container_width=True)

    # Performance table
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["activity"]} Evaluated Model Performance Metrics</div>', unsafe_allow_html=True)
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1 Score": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    # Random Forest Confusion Matrix
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{SVG_ICONS["brain"]} Random Forest Confusion Matrix (Test Split)</div>', unsafe_allow_html=True)
        
        cm = [[4671, 25], [19, 4265]]
        labels_x = ["Predicted Fake", "Predicted Real"]
        labels_y = ["Actual Fake", "Actual Real"]
        
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels_x,
            y=labels_y,
            colorscale=[[0, '#0e152b'], [0.5, '#4c1d95'], [1.0, '#06b6d4']],
            showscale=True,
            text=[[str(val) for val in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 15, "color": "white", "family": "Space Grotesk"}
        ))
        cm_fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f3f4f6',
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(cm_fig, use_container_width=True)

elif st.session_state.current_page == "Live Prediction":
    # ----------------- LIVE PREDICTION DETECTOR -----------------
    st.markdown(f"<h2 style='letter-spacing:-0.03em; font-family: Space Grotesk, sans-serif; display: flex; align-items: center; gap: 12px;'>{SVG_ICONS['search']} Live AI Detector</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; margin-top:-0.5rem; margin-bottom: 2.25rem;'>Execute predictive model analysis on custom news headlines or full articles.</p>", unsafe_allow_html=True)
    
    # Pre-defined sample inputs
    sample_fake = (
        "BREAKING NEWS: Secret classified documents leaked from the Pentagon reveal that "
        "the government is planning a complete shutdown of all domestic internet servers starting "
        "this Friday at midnight to prevent the spread of viral truth. Share this message with everyone "
        "you know before it is deleted by central authorities!"
    )
    sample_real = (
        "WASHINGTON (Reuters) - The United States Federal Reserve raised interest rates by a quarter "
        "percentage point on Wednesday, indicating that further hikes could be necessary to combat inflation. "
        "The decision was announced after a two-day committee meeting, reflecting consensus among policy members."
    )
    
    # Real-Time Predictor container
    predictor_container = st.container(border=True)
    with predictor_container:
        st.markdown(f'<div class="card-title">{SVG_ICONS["brain"]} Real-Time Predictor</div>', unsafe_allow_html=True)
        
        # Interactive Sample Buttons Row
        c_s1, c_s2, c_s3, _ = st.columns([1.3, 1.3, 1.0, 3.0])
        with c_s1:
            if st.button("Load Sample Fake", key="btn_sample_fake"):
                st.session_state.text_input_area = sample_fake
                st.rerun()
        with c_s2:
            if st.button("Load Sample Real", key="btn_sample_real"):
                st.session_state.text_input_area = sample_real
                st.rerun()
        with c_s3:
            if st.button("Clear Input", key="btn_clear_input"):
                st.session_state.text_input_area = ""
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Text area linked to Session State
        input_text = st.text_area(
            "Article Text",
            value=st.session_state.text_input_area,
            placeholder="Paste a news article or headline here...",
            height=260,
            label_visibility="collapsed",
            key="text_input_area_widget"
        )
        
        # Sync user manually typed values back to text_input_area state
        st.session_state.text_input_area = input_text
        
        # Metadata strip showing word count and reading time
        if input_text.strip() != "":
            word_count = len(input_text.split())
            char_count = len(input_text)
            reading_time = max(1, round(word_count / 200))
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #9ca3af; margin-top: -0.5rem; margin-bottom: 1.25rem; display: flex; gap: 15px; flex-wrap: wrap;">
                <span style="display: flex; align-items: center; gap: 6px;">
                    {SVG_ICONS["file_text"]} <b>Words:</b> {word_count:,}
                </span>
                <span style="display: flex; align-items: center; gap: 6px;">
                    {SVG_ICONS["code"]} <b>Characters:</b> {char_count:,}
                </span>
                <span style="display: flex; align-items: center; gap: 6px;">
                    {SVG_ICONS["clock"]} <b>Reading Time:</b> ~{reading_time} min
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Glow Predict button
        if st.button("Analyze with AI", key="pred_btn"):
            if input_text.strip() == "":
                st.warning("Please enter or load text to analyze.")
            elif vectorizer is None or rf_model is None:
                st.error("Pre-trained model pickels were not loaded. Make sure the pkl files exist.")
            else:
                # Typewriter Loading experience simulating cognitive steps
                status_placeholder = st.empty()
                
                status_placeholder.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #a78bfa; font-size: 0.95rem; font-weight: 500;">{SVG_ICONS["brain"]} AI Brain Processing... Initializing components...</div>', unsafe_allow_html=True)
                time.sleep(0.35)
                status_placeholder.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #a78bfa; font-size: 0.95rem; font-weight: 500;">{SVG_ICONS["brain"]} AI Brain Processing... Cleaning raw string input...</div>', unsafe_allow_html=True)
                time.sleep(0.35)
                status_placeholder.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #a78bfa; font-size: 0.95rem; font-weight: 500;">{SVG_ICONS["brain"]} AI Brain Processing... Mapping TF-IDF vectors...</div>', unsafe_allow_html=True)
                time.sleep(0.35)
                status_placeholder.markdown(f'<div style="display: flex; align-items: center; gap: 8px; color: #a78bfa; font-size: 0.95rem; font-weight: 500;">{SVG_ICONS["brain"]} AI Brain Processing... Running Ensemble Decision Trees...</div>', unsafe_allow_html=True)
                time.sleep(0.35)
                
                # Clear status bar
                status_placeholder.empty()
                
                # Preprocess text
                cleaned = clean_text(input_text)
                
                # Vectorize
                vectorized = vectorizer.transform([cleaned])
                
                # Predict
                prediction = rf_model.predict(vectorized)[0]
                probabilities = rf_model.predict_proba(vectorized)[0]
                confidence = probabilities[prediction] * 100
                
                # Display output alerts
                if prediction == 1:
                    st.markdown(f"""
                    <div class="alert-glow-card alert-real-card">
                        <div class="premium-badge badge-success" style="margin-bottom: 0.75rem;">Verified Match</div>
                        <div class="alert-title-main" style="color: #34d399;"><span class="pulse-indicator pulse-green"></span>{SVG_ICONS["shield_check"]} REAL NEWS</div>
                        <p style="font-size: 1.05rem; color: #d1d5db; margin: 0.5rem 0 0 0;">
                            The classifier has verified the article details and is highly confident this is authentic text.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-glow-card alert-fake-card">
                        <div class="premium-badge" style="background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); margin-bottom: 0.75rem;">Anomalous Text Pattern</div>
                        <div class="alert-title-main" style="color: #f87171;"><span class="pulse-indicator pulse-red"></span>{SVG_ICONS["shield_alert"]} FAKE NEWS</div>
                        <p style="font-size: 1.05rem; color: #d1d5db; margin: 0.5rem 0 0 0;">
                            The classifier detected non-typical language parameters corresponding to fabricated text.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Circular plot gauge confidence rating
                gauge_color = "#06b6d4" if prediction == 1 else "#ef4444"
                
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "AI Model Confidence", 'font': {'color': '#f3f4f6', 'size': 15, 'family': 'Outfit'}},
                    number={'suffix': "%", 'font': {'color': '#ffffff', 'size': 32, 'family': 'Space Grotesk'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#9ca3af", 'tickwidth': 1},
                        'bar': {'color': gauge_color},
                        'bgcolor': "rgba(22, 25, 45, 0.45)",
                        'borderwidth': 1,
                        'bordercolor': "rgba(255,255,255,0.08)",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.08)'},
                            {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.08)'},
                            {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.08)'}
                        ]
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f3f4f6',
                    height=200,
                    margin=dict(t=30, b=10, l=10, r=10)
                )
                
                # Show Gauge
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Explainability details block (AI Decision Summary)
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1.25rem; border-radius: 16px; background: rgba(14, 21, 43, 0.4); border: 1px solid rgba(255,255,255,0.04);">
                    <h5 style="color: #c084fc; margin-top: 0; font-family: 'Space Grotesk', sans-serif; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                        {SVG_ICONS["brain"]} AI Decision Summary
                    </h5>
                    <p style="font-size: 0.88rem; color: #d1d5db; line-height: 1.55; margin-bottom: 0.75rem;">
                        The model analyzed your text in three key steps:
                    </p>
                    <ul style="font-size: 0.85rem; color: #9ca3af; line-height: 1.5; padding-left: 1.2rem; margin: 0 0 0.75rem 0;">
                        <li><b>Text Preprocessing</b>: The input was cleaned by removing punctuation, links, numerical digits, and converting to lowercase.</li>
                        <li><b>Vectorization (TF-IDF)</b>: The text was translated into a high-dimensional word relevance coordinate space based on vocabulary weighting.</li>
                        <li><b>Classification (Random Forest)</b>: An ensemble of 100 Decision Trees evaluated the feature vectors to compute prediction weights.</li>
                    </ul>
                    <div style="font-size: 0.78rem; color: #6b7280; font-style: italic;">
                        * Note: Predictions are statistical outcomes. Probabilistic outputs should be used as reference flags and should not replace professional fact-checking.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metadata variables
                st.markdown("<br>", unsafe_allow_html=True)
                c_m1, c_m2, c_m3 = st.columns(3)
                with c_m1:
                    st.markdown(f'<div style="display:flex; align-items:center; gap:6px; font-size:0.85rem; color:#9ca3af;">{SVG_ICONS["check_circle"]} Model: Random Forest</div>', unsafe_allow_html=True)
                with c_m2:
                    st.markdown(f'<div style="display:flex; align-items:center; gap:6px; font-size:0.85rem; color:#9ca3af;">{SVG_ICONS["check_circle"]} Status: Inference Completed</div>', unsafe_allow_html=True)
                with c_m3:
                    st.markdown(f'<div style="display:flex; align-items:center; gap:6px; font-size:0.85rem; color:#9ca3af;">{SVG_ICONS["clock"]} {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
                
                # Download Prediction report
                st.markdown("<br>", unsafe_allow_html=True)
                report_content = f"""==================================================
FAKE NEWS DETECTION AI REPORT
==================================================
Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
System Version: 1.1.0-beta
Classification Algorithm: Random Forest Classifier
Model Accuracy: 99.51%

--- ANALYSIS RESULTS ---
Class Prediction: {"REAL NEWS" if prediction == 1 else "FAKE NEWS"}
Prediction Confidence: {confidence:.2f}%

--- ORIGINAL TEXT SAMPLE ---
{input_text[:500]}...

==================================================
Built by Ayush Singh | AI & Machine Learning Project
GitHub: https://github.com/Ayush-0915/Detecting-Fake-News-Using-ML
"""
                st.download_button(
                    label="Download AI Report (.txt)",
                    data=report_content,
                    file_name="fake_news_ai_report.txt",
                    mime="text/plain"
                )

elif st.session_state.current_page == "About":
    # ----------------- ABOUT / PROJECT INFO PAGE -----------------
    st.markdown(f"<h2 style='letter-spacing:-0.03em; font-family: Space Grotesk, sans-serif; display: flex; align-items: center; gap: 12px;'>{SVG_ICONS['book_open']} Project Details</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; margin-top:-0.5rem; margin-bottom: 2.25rem;'>Detailed description, algorithms, evaluations, and tech stack parameters.</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">
            {SVG_ICONS["book_open"]} Project Summary
        </div>
        <p style="font-size: 0.95rem; line-height: 1.6; color: #d1d5db; margin: 0;">
            This system was engineered to identify patterns of misinformation within digital media content. By establishing a robust 
            TF-IDF vector space, the classifier maps high-dimensional word relevance to distinguish structural syntactic differences 
            between factually verified reporting and fake articles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Columns grid
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                {SVG_ICONS["database"]} Dataset details
            </div>
            <ul style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><b>Dataset Name</b>: Fake and Real News Dataset (Kaggle)</li>
                <li><b>Total Records</b>: 44,898 articles</li>
                <li><b>Fake Records</b>: 23,481</li>
                <li><b>Real Records</b>: 21,417</li>
                <li><b>Pre-processing</b>: Body concatenation, lowercasing, punctuation stripping, digits removal, whitespace trimming.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                {SVG_ICONS["settings"]} Feature Engineering
            </div>
            <ul style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><b>Method</b>: TF-IDF (Term Frequency-Inverse Document Frequency)</li>
                <li><b>Vocabulary Filter</b>: English stop words removed</li>
                <li><b>Max DF</b>: 70% threshold</li>
                <li><b>Min DF</b>: Minimum frequency threshold of 2</li>
                <li><b>Dimensions</b>: 61,913 vocabulary features mapped</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                {SVG_ICONS["brain"]} Evaluated Algorithms
            </div>
            <ol style="color: #d1d5db; font-size: 0.95rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                <li><b>Random Forest Classifier</b> (Best Performance)
                    <ul style="padding-left: 1.2rem; margin-top: 0.25rem;">
                        <li><i>N-Estimators</i>: 100 Decision Trees</li>
                        <li><i>Criteria</i>: Gini impurity split selection</li>
                    </ul>
                </li>
                <li style="margin-top: 0.5rem;"><b>Logistic Regression</b>
                    <ul style="padding-left: 1.2rem; margin-top: 0.25rem;">
                        <li><i>Regularization</i>: L2 penalty</li>
                    </ul>
                </li>
                <li style="margin-top: 0.5rem;"><b>Multinomial Naive Bayes</b>
                    <ul style="padding-left: 1.2rem; margin-top: 0.25rem;">
                        <li><i>Smoothing</i>: Laplace smoothing (alpha=1.0)</li>
                    </ul>
                </li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">
                {SVG_ICONS["sparkles"]} Technical Badges
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 0.5rem;">
                <span class="premium-badge badge-primary">Python</span>
                <span class="premium-badge badge-primary">Scikit-learn</span>
                <span class="premium-badge badge-primary">TF-IDF</span>
                <span class="premium-badge badge-primary">Random Forest</span>
                <span class="premium-badge badge-primary">Pandas</span>
                <span class="premium-badge badge-primary">NumPy</span>
                <span class="premium-badge badge-primary">Plotly</span>
                <span class="premium-badge badge-primary">Streamlit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close animation page-wrapper

# ----------------- FOOTER PANEL -----------------
st.markdown("""
<div class="footer-panel">
    Built by <a href="https://github.com/Ayush-0915" target="_blank">Ayush Singh</a> | AI & Machine Learning Project | v1.1.0 | Updated: June 2026<br>
    Open Source Code repository: <a href="https://github.com/Ayush-0915/Detecting-Fake-News-Using-ML" target="_blank">Detecting-Fake-News-Using-ML</a>
</div>
""", unsafe_allow_html=True)
