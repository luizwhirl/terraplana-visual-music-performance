import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import librosa
import sys

def main():
    audio_path = "olharpratras/4-cais.mp3"
    
    try:
        y, sr = librosa.load(audio_path, sr=22050)
    except Exception as e:
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
    gluOrtho2D(-display_width/2, display_width/2, -display_height/2, display_height/2)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)

    try:
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
    except Exception as e:
        pygame.quit()
        return

    clock = pygame.time.Clock()
    running = True
    rot_global = 0.0

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
                energia_total = np.mean(mel_spect[:, quadro_atual])
                rot_global += energia_total * 3.0
                
                num_segmentos = 16
                
                for seg in range(num_segmentos):
                    angulo_base = (360.0 / num_segmentos) * seg
                    
                    glPushMatrix()
                    glRotatef(angulo_base + rot_global, 0, 0, 1)
                    
                    for i in range(0, n_mels, 4):
                        intensidade = np.mean(mel_spect[i:i+4, quadro_atual])
                        if intensidade < 0.02:
                            continue
                            
                        raio = intensidade * (display_height * 0.6) + (i * 4)
                        largura = intensidade * 35.0 * (1.0 + energia_total * 2)
                        
                        glColor4f(intensidade, intensidade, intensidade, intensidade * 0.7)
                        
                        glBegin(GL_TRIANGLES)
                        glVertex2f(0, 0)
                        glVertex2f(-largura, raio)
                        glVertex2f(largura, raio)
                        glEnd()
                        
                        glBegin(GL_POLYGON)
                        glVertex2f(0, raio)
                        glVertex2f(largura, raio + largura)
                        glVertex2f(0, raio + largura * 2.5)
                        glVertex2f(-largura, raio + largura)
                        glEnd()
                        
                        if i % 8 == 0:
                            glPushMatrix()
                            glTranslatef(0, raio, 0)
                            glRotatef(-rot_global * 3, 0, 0, 1)
                            glBegin(GL_QUADS)
                            glVertex2f(-largura, -largura)
                            glVertex2f(largura, -largura)
                            glVertex2f(largura, largura)
                            glVertex2f(-largura, largura)
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