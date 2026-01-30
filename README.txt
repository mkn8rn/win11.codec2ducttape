# Codec2DuctTape

A duct taped Codec2 vocoder with additional voice effects for fun voice transformations.

# Notice

This project is distributed under the terms of the GNU Affero General Public License v3.0 (AGPL-3.0); it dynamically links against third-party libraries (including Codec2, NAudio, and GCC runtime components) that are distributed under their respective licenses, which are included in the licenses/ directory.
This project uses Codec2 (LGPL v2.1+) https://github.com/drowe67/codec2 Built from tag v1.2.0
This project uses MinGW-w64 runtime libraries https://www.mingw-w64.org/

## Requirements

- Windows 10/11
- .NET 10.0 or later
- Virtual audio cable (e.g., VB-Audio Virtual Cable)
- codec2.dll and runtime DLLs (in binaries folder)

## Setup

1. Install VB-Audio Virtual Cable (or similar)
2. Place codec2.dll and MinGW runtime DLLs in the binaries folder
3. Edit settings.txt to configure:
   - render_device: Name of your virtual cable input (e.g., CABLE Input)
   - codec2_mode: Bitrate (3200, 2400, 1600, 1400, 1300, 1200, 700, 450)
   - sample_rate: Must be 8000 for Codec2
   - Enable/configure desired effects
4. In your target application, select CABLE Output as the microphone

## Usage

Run with "dotnet run" or launch the built executable.
