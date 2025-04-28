#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>

using namespace cv;
using namespace std;

// Estrutura que define um balão
struct Balloon {
    Point2f pos;       // Posição do balão na tela
    Point2f velocity;  // Velocidade do balão (movimento)
    bool popped = false;  // Se o balão foi estourado
    Mat sprite;       // Imagem do balão
    int points;       // Pontuação associada ao balão
    bool visible = false;  // Se o balão está visível na tela
};

// Funções para o jogo
void detectFace(Mat& frame, CascadeClassifier& cascade, vector<Rect>& faces);
void spawnBalloon(vector<Balloon>& balloons, int width, int height, const vector<pair<Mat, int>>& balloonSprites, int round);
void updateBalloons(vector<Balloon>& balloons);
void drawBalloons(Mat& frame, const vector<Balloon>& balloons);
void checkCollisions(vector<Balloon>& balloons, const vector<Rect>& faces, int& score);
void saveScore(const string& name, int score);
vector<pair<string, int>> getTopScores();
void showTopScoresOnScreen(Mat& frame, const vector<pair<string, int>>& scores);
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location);

// Constantes do jogo
string windowName = "BloonsCV6";
const int SPAWN_INTERVAL = 30;  // Intervalo para spawn de balões
float balloonSpeeds[] = {5.0, 10.0, 15.0, 20.0, 25.0};  // Velocidades dos balões
const int BALLOON_RADIUS = 30;  // Raio dos balões
bool cameraFlipped = false;  // Controle para inverter a câmera

int main() {
    // Abertura da câmera
    VideoCapture cap("/dev/video2");
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera." << endl;
        return -1;
    }

    // Carregamento do classificador Haar para detectar rostos
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Erro ao carregar haarcascade_frontalface_default.xml" << endl;
        return -1;
    }

    srand((unsigned int)time(0));  // Inicializando o gerador de números aleatórios

    Mat frame;
    vector<Balloon> balloons;  // Vetor de balões
    int frameCount = 0;  // Contador de frames
    int score = 0;  // Pontuação do jogador
    int lives = 5;  // Vidas do jogador
    int round = 1;  // Rodada atual
    string playerName = "";  // Nome do jogador

    // Inicialização da janela de exibição
    namedWindow(windowName, WINDOW_NORMAL);

    // Carregar sprites dos balões (imagens e pontuação associada)
    vector<pair<Mat, int>> balloonSprites = {
        {imread("sprites/balaoVermelho.png", IMREAD_UNCHANGED), 1},
        {imread("sprites/balaoAzul.png", IMREAD_UNCHANGED), 2},
        {imread("sprites/balaoVerde.png", IMREAD_UNCHANGED), 3},
        {imread("sprites/balaoAmarelo.png", IMREAD_UNCHANGED), 4},
        {imread("sprites/balaoRosa.png", IMREAD_UNCHANGED), 5}
    };

    // Verificar se todos os sprites foram carregados corretamente
    for (auto& sprite : balloonSprites) {
        if (sprite.first.empty()) {
            cerr << "Erro ao carregar sprites." << endl;
            return -1;
        }
    }

    // Loop principal do jogo
    while (true) {
        cap >> frame;  // Captura de frame da câmera
        if (frame.empty()) break;  // Se o frame estiver vazio, o jogo termina

        // Se a câmera estiver invertida, fazer a inversão da imagem
        if (cameraFlipped)
            flip(frame, frame, 1);

        vector<Rect> faces;  // Vetor para armazenar as faces detectadas
        detectFace(frame, face_cascade, faces);  // Detectar faces no frame

        // Lógica de avanço de rodada
        if (score >= round * 10) {  // Avança de rodada a cada múltiplo de 10 pontos
            round++;                 // Incrementa a rodada
            score = 0;               // Reseta a pontuação
        }

        // Spawna um novo balão a cada intervalo
        if (frameCount % SPAWN_INTERVAL == 0) {
            spawnBalloon(balloons, frame.cols, frame.rows, balloonSprites, round);
        }

        updateBalloons(balloons);  // Atualiza a posição dos balões
        checkCollisions(balloons, faces, score);  // Verifica colisões entre balões e rostos
      
        // Remover balões fora da tela e descontar vidas
        balloons.erase(remove_if(balloons.begin(), balloons.end(), [&](Balloon& b) {
            if (!b.popped) {
                if (b.visible && (b.pos.x < -BALLOON_RADIUS || b.pos.x > frame.cols + BALLOON_RADIUS ||
                                  b.pos.y < -BALLOON_RADIUS || b.pos.y > frame.rows + BALLOON_RADIUS)) {
                    lives--;  // Perde uma vida
                    return true;
                }
                if (b.pos.x >= 0 && b.pos.x <= frame.cols && b.pos.y >= 0 && b.pos.y <= frame.rows)
                    b.visible = true;  // Marca o balão como visível se dentro da tela
            }
            return b.popped;  // Remove o balão se estourado
        }), balloons.end());

        drawBalloons(frame, balloons);  // Desenha os balões na tela

        // Exibe a pontuação, vidas e rodada na tela
        putText(frame, "Pontuacao: " + to_string(score), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,0), 2);  // Pontuação 
        putText(frame, "Vidas: " + to_string(lives), Point(10, 70), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,255), 2);  // Vidas 
        putText(frame, "Rodada: " + to_string(round), Point(frame.cols - 200, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,0), 2);  // Rodada 

        // Desenha as faces detectadas
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(255,0,0), 2);
        }

        imshow(windowName, frame);  // Exibe o frame na janela
        frameCount++;

        // Verifica se o jogador perdeu todas as vidas
        if (lives <= 0) {
            // Fase de Game Over
            Mat gameOverFrame = frame.clone();
            putText(gameOverFrame, "GAME OVER!", Point(150, 200), FONT_HERSHEY_DUPLEX, 2.0, Scalar(0,0,255), 4);
            putText(gameOverFrame, "Digite seu nome:", Point(100, 300), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
            imshow(windowName, gameOverFrame);

            playerName = "";
            while (true) {
                int key = waitKey(0);
                if (key == 13 || key == 10) // Enter
                    break;
                else if (key == 8 && !playerName.empty())
                    playerName.pop_back();
                else if (key >= 32 && key <= 126)
                    playerName += (char)key;

                Mat nameFrame = gameOverFrame.clone();
                putText(nameFrame, playerName, Point(100, 350), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,0), 2);
                imshow(windowName, nameFrame);
            }

            saveScore(playerName, score);  // Salva o nome e a pontuação do jogador

            vector<pair<string, int>> topScores = getTopScores();  // Obtém as melhores pontuações
            Mat scoreFrame(frame.size(), CV_8UC3, Scalar(0,0,0));  // Cria uma tela para exibir o ranking
            putText(scoreFrame, "Top 5 Pontuacoes:", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
            for (size_t i = 0; i < topScores.size() && i < 5; ++i) {
                putText(scoreFrame, to_string(i+1) + ". " + topScores[i].first + " - " + to_string(topScores[i].second),
                        Point(100, 150 + 50*i), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,255), 2);
            }

            // Exibe as opções de reiniciar ou sair
            putText(scoreFrame, "Pressione ESPAÇO para jogar de novo", Point(50, frame.rows - 100),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);
            putText(scoreFrame, "Pressione ESC para sair", Point(50, frame.rows - 50),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);

            imshow(windowName, scoreFrame);

            while (true) {
                int key = waitKey(0);
                if (key == 27) return 0;  // ESC
                if (key == 32) { // Espaço
                    score = 0;
                    lives = 5;  // Reinicia as vidas
                    round = 1;  // Reinicia a rodada
                    balloons.clear();
                    frameCount = 0;
                    break;
                }
            }
        }

        char key = (char)waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        if (key == 'i' || key == 'I')
            cameraFlipped = !cameraFlipped;
    }

    return 0;
}

// Função para detectar rostos no frame
void detectFace(Mat& frame, CascadeClassifier& cascade, vector<Rect>& faces) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);  // Converte para escala de cinza
    equalizeHist(gray, gray);  // Equaliza o histograma
    cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));  // Detecta múltiplos rostos
}

// Função para gerar balões aleatórios
void spawnBalloon(vector<Balloon>& balloons, int width, int height, const vector<pair<Mat, int>>& balloonSprites, int round) {
    Balloon b;
    int side = rand() % 4;  // Define o lado da tela de onde o balão vai surgir
    int type = 0;  // Variável que vai determinar o tipo do balão

    // Lógica para determinar o tipo de balão baseado na rodada
    switch(round){
        case 1:{ type = rand() % 1; break;} // Apenas balões vermelhos na rodada 1
        case 2:{ type = rand() % 2; break;} // Apenas balões vermelhos na rodada 2
        case 3:{ type = rand() % 3; break;} // Apenas balões vermelhos na rodada 3
        case 4:{ type = rand() % 4; break;} // Apenas balões vermelhos na rodada 4
        default: { type = rand() % balloonSprites.size(); } // Todos os tipos de balões a partir da rodada 5
    }

    b.sprite = balloonSprites[type].first;
    b.points = balloonSprites[type].second;

    switch (side) {
        case 0: b.pos = Point2f(rand() % width, 0); b.velocity = Point2f(0, balloonSpeeds[type]); break;
        case 1: b.pos = Point2f(rand() % width, height); b.velocity = Point2f(0, -balloonSpeeds[type]); break;
        case 2: b.pos = Point2f(0, rand() % (height / 2) + (height / 4)); b.velocity = Point2f(balloonSpeeds[type], 0); break;
        case 3: b.pos = Point2f(width, rand() % (height / 2) + (height / 4)); b.velocity = Point2f(-balloonSpeeds[type], 0); break;
    }

    balloons.push_back(b);
}

// Atualiza as posições dos balões
void updateBalloons(vector<Balloon>& balloons) {
    for (auto& b : balloons) {
        if (!b.popped)
            b.pos += b.velocity;  // Atualiza a posição com base na velocidade
    }
}

// Função para sobrepor a imagem do balão
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location) {
    background.copyTo(output);
    for (int y = max((int)location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y;
        if (fY >= foreground.rows) break;
        for (int x = max((int)location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x;
            if (fX >= foreground.cols) break;

            double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;

            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];

                unsigned char blended = (unsigned char)(
                    foregroundPx * opacity + backgroundPx * (1. - opacity));
                output.data[y * output.step + x * output.channels() + c] = blended;
            }
        }
    }
}

// Função para desenhar os balões na tela
void drawBalloons(Mat& frame, const vector<Balloon>& balloons) {
    for (const auto& b : balloons) {
        if (b.visible) {
            Mat balloonResized;
            resize(b.sprite, balloonResized, Size(BALLOON_RADIUS * 2, BALLOON_RADIUS * 2));  // Redimensiona o balão
            overlayImage(frame, balloonResized, frame, b.pos);  // Sobrepõe o balão na tela
        }
    }
}

// Verifica se houve colisão entre balões e rostos
void checkCollisions(vector<Balloon>& balloons, const vector<Rect>& faces, int& score) {
    for (auto& b : balloons) {
        if (!b.popped && b.visible) {
            for (const auto& face : faces) {
                if (face.contains(b.pos)) {  // Se o balão colidir com a face
                    b.popped = true;  // O balão é estourado
                    score += b.points;  // Adiciona pontos
                }
            }
        }
    }
}

// Salva o nome e a pontuação do jogador
void saveScore(const string& name, int score) {
    ofstream file("scores.txt", ios::app);
    if (file.is_open()) {
        file << name << "," << score << endl;
    }
}

// Obtém as melhores pontuações
vector<pair<string, int>> getTopScores() {
    vector<pair<string, int>> scores;
    ifstream file("scores.txt");
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string name;
        int score;
        getline(ss, name, ',');
        ss >> score;
        scores.push_back(make_pair(name, score));
    }

    // Ordena as pontuações do maior para o menor
    sort(scores.begin(), scores.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
        return a.second > b.second;
    });

    return scores;
}

// Exibe as melhores pontuações na tela
void showTopScoresOnScreen(Mat& frame, const vector<pair<string, int>>& scores) {
    int y = 100;
    putText(frame, "Top 5 Pontuacoes:", Point(100, y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,255,255), 2);
    for (size_t i = 0; i < scores.size() && i < 5; ++i) {
        putText(frame, to_string(i+1) + ". " + scores[i].first + " - " + to_string(scores[i].second),
                Point(100, y + 50 * (i + 1)), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,255,255), 2);
    }
}
