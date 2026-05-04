import os
import glob
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import librosa
import sys

def selecionar_musica(display_width, display_height):
    pygame.init()
    tela = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Selecao de Musica")
    
    fonte_titulo = pygame.font.SysFont("Arial", 36, bold=True)
    fonte_item = pygame.font.SysFont("Arial", 28)
    
    arquivos = glob.glob("olharpratras/*.mp3")
    if not arquivos:
        arquivos = glob.glob("*.mp3")
        
    if not arquivos:
        sys.exit()

    selecionado = 0
    rodando = True
    clock = pygame.time.Clock()

    while rodando:
        tela.fill((20, 20, 25))
        
        titulo = fonte_titulo.render("Selecione a Musica", True, (255, 255, 255))
        tela.blit(titulo, (display_width//2 - titulo.get_width()//2, 50))

        for i, arq in enumerate(arquivos):
            nome_arq = os.path.basename(arq)
            if i == selecionado:
                cor = (50, 255, 150)
                prefixo = ">>  "
            else:
                cor = (150, 150, 150)
                prefixo = "    "
                
            texto = fonte_item.render(prefixo + nome_arq, True, cor)
            tela.blit(texto, (display_width//2 - 200, 150 + i * 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selecionado = (selecionado - 1) % len(arquivos)
                elif event.key == K_DOWN:
                    selecionado = (selecionado + 1) % len(arquivos)
                elif event.key == K_RETURN:
                    rodando = False

        clock.tick(30)

    tela.fill((20, 20, 25))
    nome_musica = os.path.basename(arquivos[selecionado])
    texto_carga = fonte_titulo.render(f"Analisando: {nome_musica}...", True, (255, 200, 50))
    texto_aviso = fonte_item.render("Aguarde...", True, (200, 200, 200))
    tela.blit(texto_carga, (display_width//2 - texto_carga.get_width()//2, display_height//2 - 30))
    tela.blit(texto_aviso, (display_width//2 - texto_aviso.get_width()//2, display_height//2 + 20))
    pygame.display.flip()

    return arquivos[selecionado]

def main():
    display_width, display_height = 800, 600
    audio_path = selecionar_musica(display_width, display_height)
    
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

    pygame.mixer.init()
    pygame.display.set_mode((display_width, display_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(os.path.basename(audio_path))

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