import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import librosa
import sys

def main():
    audio_path = "olharpratras/4-cais.mp3"
    
    print("Analysing audio file... This may take a while")
    
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
        print(f"Path error: {e}")
        return

    hop_length = 512
    n_mels = 64
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    mel_spect = (mel_spect + 80) / 80
    mel_spect = np.clip(mel_spect, 0, 1)

    fps_audio = sr / hop_length

    pygame.init()
    pygame.mixer.init()
    display_width, display_height = 800, 600
    pygame.display.set_mode((display_width, display_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("vou me jogar nesse mar de lágrimas")

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, display_width, 0, display_height)
    glMatrixMode(GL_MODELVIEW)

    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Pygame music error: {e}")
        pygame.quit()
        return

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        tempo_atual_ms = pygame.mixer.music.get_pos()
        
        if tempo_atual_ms != -1:
            tempo_atual_segundos = tempo_atual_ms / 1000.0
            quadro_atual = int(tempo_atual_segundos * fps_audio)

            if quadro_atual < mel_spect.shape[1]:
                largura_barra = display_width / n_mels
                
                for i in range(n_mels):
                    intensidade = mel_spect[i, quadro_atual]
                    altura_barra = intensidade * display_height * 0.9
                    
                    glColor3f(intensidade, intensidade, intensidade)
                    
                    x1 = i * largura_barra
                    x2 = x1 + (largura_barra * 0.8)
                    y1 = 0
                    y2 = altura_barra

                    glBegin(GL_QUADS)
                    glVertex2f(x1, y1)
                    glVertex2f(x2, y1)
                    glVertex2f(x2, y2)
                    glVertex2f(x1, y2)
                    glEnd()

        pygame.display.flip()
        clock.tick(60)
        
        if not pygame.mixer.music.get_busy() and tempo_atual_ms > 0:
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()