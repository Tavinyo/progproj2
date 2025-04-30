#include "opencv2/opencv.hpp"
#include <SFML/Audio.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <map>
#include <ctime>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace sf;

// Estrutura que define um balão
struct Balloon {
    Point2f pos;            // Posição do balão na tela
    Point2f velocity;       // Velocidade do balão (movimento)
    bool popped = false;    // Se o balão foi estourado
    Mat sprite;            // Imagem do balão
    int points;            // Pontuação associada ao balão
    bool visible = false;  // Se o balão está visível na tela
    int framesSincePop = 0; // Frames desde que foi estourado (para animação)
};

// Constantes do jogo
const string windowName = "BloonsCV6";
const int WIDTH = 800;
const int HEIGHT = 600;
const int SPAWN_INTERVAL = 30;        // Intervalo para spawn de balões
const int BALLOON_RADIUS = 30;        // Raio dos balões
const int INITIAL_LIVES = 5;          // Vidas iniciais
const int ROUND_POINTS_MULTIPLIER = 10; // Pontos necessários por rodada

// Variáveis globais
bool cameraFlipped = true;
map<int, Mat> dardoCache; // Cache de dardos redimensionados
Mat popAnimation;         // Animação de estouro de balão
SoundBuffer popBuffer, gameOverBuffer, roundStartBuffer;
Sound popSound, gameOverSound, roundStartSound;

// Protótipos de função
int mostrarMenuPrincipal();
void showTopScores();
void initGameResources(vector<pair<Mat, int>>& balloonSprites, Mat& background, Mat& dardo);
void gameLoop(VideoCapture& cap, CascadeClassifier& face_cascade);
void detectFaces(Mat& cameraFrame, CascadeClassifier& cascade, vector<Rect>& faces);
void spawnBalloon(vector<Balloon>& balloons, int width, int height, const vector<pair<Mat, int>>& balloonSprites, int round);
void updateBalloons(vector<Balloon>& balloons);
void drawGameFrame(Mat& frame, const vector<Balloon>& balloons, const vector<Rect>& faces, 
                  int score, int lives, int round, const Mat& background, const Mat& dardo);
void checkCollisions(vector<Balloon>& balloons, const vector<Rect>& faces, int& score);
void gameOverScreen(int& score, int& round, vector<Balloon>& balloons, int& frameCount);
void saveScore(const string& name, int score);
vector<pair<string, int>> getTopScores();
void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location);
void carregarSons();


int main() {
    // Carregar sons
    carregarSons();

    // Inicialização
    srand((unsigned int)time(0));
    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, WIDTH, HEIGHT);

    // Carregar recursos
    vector<pair<Mat, int>> balloonSprites;
    Mat background, dardo;
    initGameResources(balloonSprites, background, dardo);

    // Menu principal
    int escolha = mostrarMenuPrincipal();
    while (escolha != 0) {
        if (escolha == 2) {
            showTopScores();
            escolha = mostrarMenuPrincipal();
            continue;
        }

        // Inicializar câmera e detector de faces
        VideoCapture cap("/dev/video0");
        if (!cap.isOpened()) {
            cerr << "Erro ao abrir a câmera." << endl;
            return -1;
        }

        CascadeClassifier face_cascade;
        if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
            cerr << "Erro ao carregar haarcascade_frontalface_default.xml" << endl;
            return -1;
        }

        // Loop principal do jogo
        gameLoop(cap, face_cascade);
        
        escolha = mostrarMenuPrincipal();
    }

    return 0;
}

// Implementações das funções

void carregarSons(){
    if (!popBuffer.loadFromFile("sons/pop.mp3")) {
        cerr << "Erro ao carregar pop.mp3\n";
    }
    if (!gameOverBuffer.loadFromFile("sons/gameover.wav")) {
        cerr << "Erro ao carregar gameover.wav\n";
    }
    if (!roundStartBuffer.loadFromFile("sons/round_start.wav")) {
        cerr << "Erro ao carregar round_start.wav\n";
    }

    popSound.setBuffer(popBuffer);
    gameOverSound.setBuffer(gameOverBuffer);
    roundStartSound.setBuffer(roundStartBuffer);
}

int mostrarMenuPrincipal() {
    Mat menuFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    putText(menuFrame, "BloonsCV6", Point(WIDTH/4, 100), FONT_HERSHEY_DUPLEX, 2.0, Scalar(0, 255, 255), 3);
    putText(menuFrame, "1 - Jogar", Point(WIDTH/4, 200), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    putText(menuFrame, "2 - Ver Top 5", Point(WIDTH/4, 300), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    putText(menuFrame, "ESC - Sair", Point(WIDTH/4, 400), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    imshow(windowName, menuFrame);

    while (true) {
        int key = waitKey(0);
        if (key == 49) return 1; // '1'
        if (key == 50) return 2; // '2'
        if (key == 27) return 0; // ESC
    }
}

void showTopScores() {
    vector<pair<string, int>> topScores = getTopScores();
    Mat topFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    
    putText(topFrame, "Top 5 Pontuacoes:", Point(100, 80), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 0), 2);
    
    for (size_t i = 0; i < topScores.size() && i < 5; ++i) {
        ostringstream oss;
        oss << (i+1) << ". " << topScores[i].first << " - " << topScores[i].second;
        putText(topFrame, oss.str(), Point(150, 150 + 50*i), 
               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
    }
    
    putText(topFrame, "Pressione qualquer tecla para voltar...", 
           Point(100, HEIGHT-50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    imshow(windowName, topFrame);
    waitKey(0);
}

void initGameResources(vector<pair<Mat, int>>& balloonSprites, Mat& background, Mat& dardo) {
    // Carregar sprites dos balões
    vector<string> balloonFiles = {
        "sprites/balaoVermelho.png", "sprites/balaoAzul.png", 
        "sprites/balaoVerde.png", "sprites/balaoAmarelo.png", 
        "sprites/balaoRosa.png"
    };
    
    vector<int> balloonPoints = {1, 2, 3, 4, 5};
    
    for (size_t i = 0; i < balloonFiles.size(); ++i) {
        Mat sprite = imread(balloonFiles[i], IMREAD_UNCHANGED);
        if (sprite.empty()) {
            cerr << "Erro ao carregar sprite: " << balloonFiles[i] << endl;
            exit(-1);
        }
        balloonSprites.push_back({sprite, balloonPoints[i]});
    }

    // Carregar fundo e dardo
    background = imread("sprites/mapa.png", IMREAD_COLOR);
    dardo = imread("sprites/dardo.png", IMREAD_UNCHANGED);
    popAnimation = imread("sprites/pop.png", IMREAD_UNCHANGED); // Nova animação de estouro
    
    if (background.empty() || dardo.empty() || popAnimation.empty()) {
        cerr << "Erro ao carregar recursos gráficos." << endl;
        exit(-1);
    }
    
    resize(background, background, Size(WIDTH, HEIGHT));
}

void gameLoop(VideoCapture& cap, CascadeClassifier& face_cascade) {
    Mat frame;
    vector<Balloon> balloons;
    int frameCount = 0, score = 0, lives = INITIAL_LIVES, round = 1;
    
    // Carregar recursos
    vector<pair<Mat, int>> balloonSprites;
    Mat background, dardo;
    initGameResources(balloonSprites, background, dardo);

    while (true) {
        Mat cameraFrame;
        cap >> cameraFrame;
        if (cameraFrame.empty()) break;

        // Detecção de rostos
        vector<Rect> faces;
        detectFaces(cameraFrame, face_cascade, faces);

        // Lógica do jogo
        if (frameCount % SPAWN_INTERVAL == 0) {
            spawnBalloon(balloons, WIDTH, HEIGHT, balloonSprites, round);
        }

        updateBalloons(balloons);
        checkCollisions(balloons, faces, score);

        // Verificar balões fora da tela
        balloons.erase(remove_if(balloons.begin(), balloons.end(), [&](Balloon& b) {
            if (!b.popped) {
                if (b.visible && (b.pos.x < -BALLOON_RADIUS || b.pos.x > WIDTH + BALLOON_RADIUS ||
                                 b.pos.y < -BALLOON_RADIUS || b.pos.y > HEIGHT + BALLOON_RADIUS)) {
                    lives--;
                    return true;
                }
                if (b.pos.x >= 0 && b.pos.x <= WIDTH && b.pos.y >= 0 && b.pos.y <= HEIGHT) {
                    b.visible = true;
                }
            }
            return b.popped && b.framesSincePop > 10; // Dar tempo para animação de estouro
        }), balloons.end());

        // Avançar rodada
        if (score >= round * ROUND_POINTS_MULTIPLIER) {
            round++;
            roundStartSound.play();
        }

        // Desenhar frame
        drawGameFrame(frame, balloons, faces, score, lives, round, background, dardo);

        // Verificar game over
        if (lives <= 0) {
            gameOverSound.play();
            gameOverScreen(score, round, balloons, frameCount);
            lives = INITIAL_LIVES; // Reset para novo jogo
        }

        // Controles
        char key = (char)waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q'){ // ESC ou Q para sair
            // Perguntar se deseja salvar a pontuação
            Mat prompt = Mat::zeros(200, 640, CV_8UC3);
            putText(prompt, "Deseja salvar sua pontuacao? (S/N)", Point(30, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            imshow("BloonsCV6", prompt);

            // Espera por S ou N
            while (true) {
                char response = (char)waitKey(0);
                if (response == 's' || response == 'S') {
                    gameOverSound.play();
                    gameOverScreen(score, round, balloons, frameCount);
                    lives = INITIAL_LIVES; // Reset para novo jogo
                    break;
                } else if (response == 'n' || response == 'N') {
                    break;
                }
            }
            break;
        }
        if (key == 'i' || key == 'I') cameraFlipped = !cameraFlipped;
        
        frameCount++;
    }
}

void detectFaces(Mat& cameraFrame, CascadeClassifier& cascade, vector<Rect>& faces) {
    Mat cameraGray;
    cvtColor(cameraFrame, cameraGray, COLOR_BGR2GRAY);
    equalizeHist(cameraGray, cameraGray);

    if (cameraFlipped) {
        flip(cameraGray, cameraGray, 1);
        flip(cameraFrame, cameraFrame, 1);
    }

    cascade.detectMultiScale(cameraGray, faces, 1.1, 3, 0, Size(30, 30));

    // Escalar rostos para o tamanho da janela
    float scaleX = (float)WIDTH / cameraFrame.cols;
    float scaleY = (float)HEIGHT / cameraFrame.rows;

    for (auto& face : faces) {
        face.x = (int)(face.x * scaleX);
        face.y = (int)(face.y * scaleY);
        face.width = (int)(face.width * scaleX);
        face.height = (int)(face.height * scaleY);
    }
}

void spawnBalloon(vector<Balloon>& balloons, int width, int height, 
                 const vector<pair<Mat, int>>& balloonSprites, int round) {
    Balloon b;
    int side = rand() % 4;
    
    // Sistema de dificuldade progressiva
    int maxType = min(round - 1, (int)balloonSprites.size() - 1);
    int type = rand() % (maxType + 1);
    
    // Velocidade baseada no tipo e rodada
    float speedMultiplier = 1.0f + (round * 0.1f);
    float speed = (5.0f + (type * 5.0f)) * speedMultiplier;
    
    b.sprite = balloonSprites[type].first;
    b.points = balloonSprites[type].second;

    // Posição e direção baseada no lado da tela
    switch (side) {
        case 0: // Topo
            b.pos = Point2f(rand() % width, -BALLOON_RADIUS);
            b.velocity = Point2f((rand() % 3 - 1) * 0.5f, speed); // Pequeno movimento horizontal
            break;
        case 1: // Base
            b.pos = Point2f(rand() % width, height + BALLOON_RADIUS);
            b.velocity = Point2f((rand() % 3 - 1) * 0.5f, -speed);
            break;
        case 2: // Esquerda
            b.pos = Point2f(-BALLOON_RADIUS, rand() % (height / 2) + (height / 4));
            b.velocity = Point2f(speed, (rand() % 3 - 1) * 0.5f);
            break;
        case 3: // Direita
            b.pos = Point2f(width + BALLOON_RADIUS, rand() % (height / 2) + (height / 4));
            b.velocity = Point2f(-speed, (rand() % 3 - 1) * 0.5f);
            break;
    }

    balloons.push_back(b);
}

void updateBalloons(vector<Balloon>& balloons) {
    for (auto& b : balloons) {
        if (!b.popped) {
            b.pos += b.velocity;
        } else {
            b.framesSincePop++;
        }
    }
}

void drawGameFrame(Mat& frame, const vector<Balloon>& balloons, const vector<Rect>& faces, 
                  int score, int lives, int round, const Mat& background, const Mat& dardo) {
    // Fundo
    background.copyTo(frame);

    // Desenhar balões
    for (const auto& b : balloons) {
        if (b.visible) {
            if (b.popped && b.framesSincePop <= 10) {
                // Animação de estouro
                Mat resizedPop;
                resize(popAnimation, resizedPop, Size(BALLOON_RADIUS * 2, BALLOON_RADIUS * 2));
                overlayImage(frame, resizedPop, frame, b.pos - Point2f(BALLOON_RADIUS, BALLOON_RADIUS));
            } else if (!b.popped) {
                // Balão normal
                Mat balloonResized;
                resize(b.sprite, balloonResized, Size(BALLOON_RADIUS * 2, BALLOON_RADIUS * 2));
                overlayImage(frame, balloonResized, frame, b.pos - Point2f(BALLOON_RADIUS, BALLOON_RADIUS));
            }
        }
    }

    // Desenhar dardos (rostos)
    for (const auto& face : faces) {
        if (dardoCache.find(face.width) == dardoCache.end()) {
            resize(dardo, dardoCache[face.width], Size(face.width, face.height));
        }
        overlayImage(frame, dardoCache[face.width], frame, Point2f(face.x, face.y));
    }

    // UI - Pontuação, vidas e rodada
    putText(frame, "Pontuacao: " + to_string(score), Point(10, 30), 
           FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(frame, "Vidas: " + to_string(lives), Point(10, 70), 
           FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    putText(frame, "Rodada: " + to_string(round), Point(WIDTH - 200, 30), 
           FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);

    imshow(windowName, frame);
}

void checkCollisions(vector<Balloon>& balloons, const vector<Rect>& faces, int& score) {
    for (auto& b : balloons) {
        if (!b.popped && b.visible) {
            for (const auto& face : faces) {
                Rect balloonRect(b.pos.x - BALLOON_RADIUS, b.pos.y - BALLOON_RADIUS, 
                               BALLOON_RADIUS * 2, BALLOON_RADIUS * 2);
                if ((balloonRect & face).area() > 0) {
                    b.popped = true;
                    score += b.points;
                    popSound.play();
                }
            }
        }
    }
}

void gameOverScreen(int& score, int& round, vector<Balloon>& balloons, int& frameCount) {
    
    // Tela de game over
    Mat gameOverFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    putText(gameOverFrame, "GAME OVER!", Point(WIDTH/4, 150), 
           FONT_HERSHEY_DUPLEX, 2.0, Scalar(0, 0, 255), 4);
    putText(gameOverFrame, "Pontuacao Final: " + to_string(score), 
           Point(WIDTH/4, 220), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(gameOverFrame, "Digite seu nome:", Point(WIDTH/4, 300), 
           FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    
    string playerName = "";
    imshow(windowName, gameOverFrame);
    
    // Capturar nome do jogador
    while (true) {
        int key = waitKey(0);
        if (key == 13 || key == 10) break; // Enter
        if (key == 8 && !playerName.empty()) playerName.pop_back(); // Backspace
        else if (key >= 32 && key <= 126) playerName += (char)key;
        
        Mat nameFrame = gameOverFrame.clone();
        putText(nameFrame, playerName, Point(WIDTH/4, 350), 
               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        imshow(windowName, nameFrame);
    }
    
    // Salvar pontuação
    saveScore(playerName, score);
    
    // Mostrar ranking
    showTopScores();
    
    // Resetar jogo
    score = 0;
    round = 1;
    balloons.clear();
    frameCount = 0;
}

void saveScore(const string& playerName, int score) {
    vector<pair<string, int>> scores;

    // Lê os scores existentes
    ifstream infile("scores.txt");
    string line;
    while (getline(infile, line)) {
        size_t sep = line.find('|');
        if (sep != string::npos) {
            string name = line.substr(0, sep);
            int sc = stoi(line.substr(sep + 1));
            scores.emplace_back(name, sc);
        }
    }
    infile.close();

    // Adiciona nova pontuação
    scores.emplace_back(playerName, score);

    // Ordena e mantém top 5
    sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });
    if (scores.size() > 5)
        scores.resize(5);

    // Salva no arquivo
    ofstream outfile("scores.txt", ios::trunc);
    for (const auto& s : scores) {
        outfile << s.first << "|" << s.second << endl;
    }
    outfile.close();
}

vector<pair<string, int>> getTopScores() {
    vector<pair<string, int>> scores;
    ifstream file("scores.txt");
    string line;

    while (getline(file, line)) {
        size_t sep = line.find('|');
        if (sep != string::npos) {
            string name = line.substr(0, sep);
            int score = stoi(line.substr(sep + 1));
            scores.push_back({name, score});
        }
    }

    sort(scores.begin(), scores.end(), [](const pair<string, int>& a, const pair<string, int>& b) {
        return a.second > b.second;
    });

    return scores;
}


void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location) {
    background.copyTo(output);
    
    for (int y = max((int)location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y;
        if (fY >= foreground.rows) break;
        
        for (int x = max((int)location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x;
            if (fX >= foreground.cols) break;
            
            double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.0;
            
            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + x * output.channels() + c] = 
                    (unsigned char)(foregroundPx * opacity + backgroundPx * (1.0 - opacity));
            }
        }
    }
}
