# 🎨 AI Image Synthesis Web App

A web application for generating images from text prompts using **Stable Diffusion v1.5**, built with **Streamlit** and **Hugging Face Diffusers**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📸 Demo

> _Add a screenshot or GIF of the app here_
>
> `![App Screenshot](assets/demo.png)`

## ✨ Features

- Generate images from natural language text prompts
- Adjustable image resolution (height/width)
- Configurable inference steps and guidance scale
- Optional negative prompts
- Optional seed input for reproducible results
- Download generated images as PNG

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web interface
- **PyTorch** — deep learning framework
- **Hugging Face Diffusers** — Stable Diffusion pipeline
- **Transformers / Accelerate / Safetensors** — model loading and optimization

## 🧠 Model

Uses [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) from the Hugging Face Hub.

## 📁 Project Structure
