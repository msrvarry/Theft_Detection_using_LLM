﻿# Theft Detection using LLM
This project performs automated theft detection by analyzing surveillance video frames using a hybrid approach:

YOLOv8 detects visible objects.

Gemini Pro Vision (1.5 Flash) provides contextual understanding of human actions in suspicious frames.

It flags potential theft scenarios such as stealing, grabbing, or hiding objects, even when object detection alone is insufficient.

# Features
Supports any pre-recorded surveillance video input

Fast object detection using YOLOv8

Scene-level semantic understanding with Google Gemini AI

Detects suspicious behavior like:

Theft

Hiding or grabbing objects

Robbery

Analyzes every Nth frame (configurable) for efficiency

Real-time alert overlay when theft is detected

Automatically saves suspicious frames for audit trail

# How It Works
YOLOv8 detects objects in each frame with a confidence threshold.

If no confident object is found, the frame is flagged as suspicious.

Every Nth suspicious frame (e.g., every 5th frame):

The frame is saved temporarily.

Sent to Gemini Vision with a specific prompt asking about theft or suspicious activity.

Gemini's response is parsed:

If it contains terms like "theft", "robbery", "stealing", or "suspicious", an alert is triggered.

Otherwise, the frame is considered safe.

The system overlays a 🚨 THEFT DETECTED 🚨 message if theft is identified.

