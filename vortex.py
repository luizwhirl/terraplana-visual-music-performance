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

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
                
                glPushMatrix()
                glTranslatef(display_width / 2, display_height / 2, 0)
                
                glRotatef(tempo_atual_segundos * 10.0, 0, 0, 1)
                
                for i in range(n_mels):
                    intensidade = mel_spect[i, quadro_atual]
                    
                    if intensidade < 0.02:
                        continue
                    
                    tom_cinza = 0.1 + (i / n_mels) * 0.4 + (intensidade * 0.5)
                    tom_cinza = np.clip(tom_cinza, 0.0, 1.0)
                    
                    alpha = 0.1 + (intensidade * 0.6)
                    glColor4f(tom_cinza, tom_cinza, tom_cinza, alpha)
                    
                    glPushMatrix()
                    
                    rotacao_banda = i * (360.0 / n_mels) + tempo_atual_segundos * (15 + intensidade * 80)
                    glRotatef(rotacao_banda, 0, 0, 1)
                    
                    distancia_do_centro = i * 6 + np.sin(tempo_atual_segundos * 1.5 + i * 0.2) * 40
                    glTranslatef(distancia_do_centro, 0, 0)
                    
                    tamanho = 15 + intensidade * 120 * (1.0 + (i / n_mels))
                    
                    glBegin(GL_QUADS)
                    glVertex2f(0, tamanho)
                    glVertex2f(tamanho * 0.4, 0)
                    glVertex2f(0, -tamanho)
                    glVertex2f(-tamanho * 0.4, 0)
                    glEnd()
                    
                    tom_inverso = 1.0 - tom_cinza
                    glColor4f(tom_inverso, tom_inverso, tom_inverso, alpha * 0.7)
                    glBegin(GL_TRIANGLES)
                    glVertex2f(0, tamanho * 0.5)
                    glVertex2f(tamanho * 0.25, -tamanho * 0.2)
                    glVertex2f(-tamanho * 0.25, -tamanho * 0.2)
                    glEnd()
                    
                    glPopMatrix()

                glPopMatrix()

        pygame.display.flip()
        clock.tick(60)
        
        if not pygame.mixer.music.get_busy() and tempo_atual_ms > 0:
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()