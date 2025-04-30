#include "opencv2/opencv.hpp"
#include <SFML/Audio.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> // Para std::remove_if, std::begin, std::end
#include <sstream>
#include <map>
#include <ctime>
#include <cstdlib>

using namespace cv;
using namespace std;
using namespace sf;

// Constantes do jogo
const string WINDOW_NAME = "BloonsCV6";
const int WIDTH = 800;
const int HEIGHT = 600;
const int SPAWN_INTERVAL = 30;
const int BALLOON_RADIUS = 30;
const int INITIAL_LIVES = 5;
const int ROUND_POINTS_MULTIPLIER = 10;
const int POP_ANIMATION_FRAMES = 10;

class Game; // Forward declaration
class GameState;
class MenuState;
class PlayState;
class GameOverState;

void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location);

class ResourceManager {
private:
    vector<pair<Mat, int>> balloonSprites;
    Mat background;
    Mat dardo;
    Mat popAnimation;

public:
    ResourceManager() = default;

    bool loadResources() {
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
                return false;
            }
            balloonSprites.push_back({sprite, balloonPoints[i]});
        }

        background = imread("sprites/mapa.png", IMREAD_COLOR);
        dardo = imread("sprites/dardo.png", IMREAD_UNCHANGED);
        popAnimation = imread("sprites/pop.png", IMREAD_UNCHANGED);

        if (background.empty() || dardo.empty() || popAnimation.empty()) {
            cerr << "Erro ao carregar recursos gráficos." << endl;
            return false;
        }
        resize(background, background, Size(WIDTH, HEIGHT));
        return true;
    }

    Mat getBalloonSprite(int index) const {
        if (index >= 0 && index < balloonSprites.size()) {
            return balloonSprites[index].first;
        }
        return Mat();
    }

    int getBalloonPoints(int index) const {
        if (index >= 0 && index < balloonSprites.size()) {
            return balloonSprites[index].second;
        }
        return 0;
    }

    Mat getBackground() const {
        return background;
    }

    Mat getDardo() const {
        return dardo;
    }

    Mat getPopAnimation() const {
        return popAnimation;
    }

    int getTotalBalloonTypes() const {
        return balloonSprites.size();
    }
};

class SoundManager {
private:
    SoundBuffer popBuffer;
    SoundBuffer gameOverBuffer;
    SoundBuffer roundStartBuffer;
    Sound popSound;
    Sound gameOverSound;
    Sound roundStartSound;

public:
    SoundManager() = default;

    bool loadSounds() {
        if (!popBuffer.loadFromFile("sons/pop.mp3")) cerr << "Erro ao carregar pop.mp3\n";
        if (!gameOverBuffer.loadFromFile("sons/gameover.mp3")) cerr << "Erro ao carregar gameover.mp3\n";
        if (!roundStartBuffer.loadFromFile("sons/round_start.mp3")) cerr << "Erro ao carregar round_start.mp3\n";

        popSound.setBuffer(popBuffer);
        gameOverSound.setBuffer(gameOverBuffer);
        roundStartSound.setBuffer(roundStartBuffer);
        return true;
    }

    void playPopSound() {
        popSound.play();
    }

    void playGameOverSound() {
        gameOverSound.play();
    }

    void playRoundStartSound() {
        roundStartSound.play();
    }
};

class ScoreManager {
private:
    vector<pair<string, int>> scoresList;
    const string SCORES_FILE = "scores.txt";

public:
    ScoreManager() {
        loadScores();
    }

    void loadScores() {
        scoresList.clear();
        ifstream file(SCORES_FILE);
        string line;
        while (getline(file, line)) {
            size_t sep = line.find('|');
            if (sep != string::npos) {
                string name = line.substr(0, sep);
                int score = stoi(line.substr(sep + 1));
                scoresList.push_back({name, score});
            }
        }
        sort(scoresList.begin(), scoresList.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        if (scoresList.size() > 5) {
            scoresList.resize(5);
        }
    }

    void saveScore(const string& name, int score) {
        scoresList.push_back({name, score});
        sort(scoresList.begin(), scoresList.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        if (scoresList.size() > 5) {
            scoresList.resize(5);
        }
        ofstream file(SCORES_FILE, ios::trunc);
        for (const auto& s : scoresList) {
            file << s.first << "|" << s.second << endl;
        }
    }

    vector<pair<string, int>> getTopScores() const {
        return scoresList;
    }
};

class FaceDetector {
private:
    CascadeClassifier cascade;
    bool cameraFlipped;
    VideoCapture capture;

public:
    FaceDetector() : cameraFlipped(true) {}

    bool init(const string& cascadeFile, const string& cameraDevice = "/dev/video0") {
        if (!cascade.load(cascadeFile)) {
            cerr << "Erro ao carregar " << cascadeFile << endl;
            return false;
        }
        if (!capture.open(cameraDevice)) {
            cerr << "Erro ao abrir a câmera." << endl;
            return false;
        }
        capture.set(CAP_PROP_FRAME_WIDTH, 320); // Tenta definir largura para 320 
        capture.set(CAP_PROP_FRAME_HEIGHT, 240); // Tenta definir altura para 240
        return true;
    }

    void detectFaces(Mat& frame, vector<Rect>& faces, int width, int height) {
        Mat grayFrame;
        capture >> frame;
        if (frame.empty()) return;

        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);

        if (cameraFlipped) {
            flip(grayFrame, grayFrame, 1);
            flip(frame, frame, 1);
        }

        cascade.detectMultiScale(grayFrame, faces, 1.1, 3, 0, Size(30, 30));

        float scaleX = (float)width / frame.cols;
        float scaleY = (float)height / frame.rows;

        for (auto& face : faces) {
            face.x = static_cast<int>(face.x * scaleX);
            face.y = static_cast<int>(face.y * scaleY);
            face.width = static_cast<int>(face.width * scaleX);
            face.height = static_cast<int>(face.height * scaleY);
        }
    }

    void toggleCameraFlip() {
        cameraFlipped = !cameraFlipped;
    }

    bool getCameraFrame(Mat& frame) {
        return capture.read(frame);
    }

    void releaseCapture() {
        capture.release();
    }
};

class Balloon {
public:
    Point2f position;
    Point2f velocity;
    bool popped;
    Mat sprite;
    int points;
    bool visible;
    int framesSincePop;

    Balloon(Point2f position, Point2f velocity, Mat sprite, int points) :
        position(position), velocity(velocity), popped(false), sprite(sprite), points(points), visible(false), framesSincePop(0) {}

    void update() {
        if (!popped) {
            position += velocity;
        } else {
            framesSincePop++;
        }
    }

    void render(Mat& frame, const Mat& popAnimation) const {
        if (visible) {
            if (popped && framesSincePop <= POP_ANIMATION_FRAMES) {
                Mat resizedPop;
                resize(popAnimation, resizedPop, Size(BALLOON_RADIUS * 2, BALLOON_RADIUS * 2));
                overlayImage(frame, resizedPop, frame, position - Point2f(BALLOON_RADIUS, BALLOON_RADIUS));
            } else if (!popped) {
                Mat balloonResized;
                resize(sprite, balloonResized, Size(BALLOON_RADIUS * 2, BALLOON_RADIUS * 2));
                overlayImage(frame, balloonResized, frame, position - Point2f(BALLOON_RADIUS, BALLOON_RADIUS));
            }
        }
    }

    bool checkCollision(const Rect& rect) const {
        return (Rect(position.x - BALLOON_RADIUS, position.y - BALLOON_RADIUS, BALLOON_RADIUS * 2, BALLOON_RADIUS * 2) & rect).area() > 0;
    }

    bool isOffScreen(int width, int height) const {
        return (position.x < -BALLOON_RADIUS || position.x > width + BALLOON_RADIUS ||
                position.y < -BALLOON_RADIUS || position.y > height + BALLOON_RADIUS);
    }

    bool getPoppedState() const {
        return popped;
    }

    void setPoppedState(bool popped) {
        this->popped = popped;
    }

    int getPoints() const {
        return points;
    }

    void setVisible(bool visible) {
        this->visible = visible;
    }

    bool isVisible() const {
        return visible;
    }

    int getFramesSincePop() const {
        return framesSincePop;
    }
};

class GameState {
protected:
    Game* game;
public:
    GameState(Game* game) : game(game) {}
    virtual void handleInput() = 0;
    virtual void update() = 0;
    virtual void render(Mat& frame) = 0;
    virtual ~GameState() = default;
};

class MenuState : public GameState {
private:
    ScoreManager* scoreManager;

public:
    MenuState(Game* game) : GameState(game), scoreManager(new ScoreManager()) {}
    ~MenuState() override { delete scoreManager; }

    void handleInput();
    void update();
    void render(Mat& frame);

private:
    void showTopScores();
};

class PlayState : public GameState {
private:
    vector<Balloon> balloons;
    vector<Rect> faces;
    int frameCount;
    ResourceManager* resources;
    FaceDetector* faceDetector;
    ScoreManager* scoreManager;
    SoundManager* soundManager;
    map<int, Mat> dardoCache;

public:
    PlayState(Game* game) : GameState(game), frameCount(0),
                           resources(new ResourceManager()),
                           faceDetector(new FaceDetector()),
                           scoreManager(new ScoreManager()),
                           soundManager(new SoundManager()) {
        resources->loadResources();
        faceDetector->init("haarcascade_frontalface_default.xml");
        soundManager->loadSounds();
    }
    ~PlayState() override {
        delete resources;
        delete faceDetector;
        delete scoreManager;
        delete soundManager;
    }

    void handleInput();
    void update();
    void render(Mat& frame);

private:
    void spawnBalloon();
    void updateBalloons();
    void checkCollisions();
    void removeOffScreenBalloons();
    void drawGameFrame(Mat& frame);
};

class GameOverState : public GameState {
private:
    string playerName;
    ScoreManager* scoreManager;
    SoundManager* soundManager;

public:
    GameOverState(Game* game) : GameState(game), scoreManager(new ScoreManager()), soundManager(new SoundManager()) {
        soundManager->loadSounds();
    }
    ~GameOverState() override {
        delete scoreManager;
        delete soundManager;
    }

    void handleInput();
    void update();
    void render(Mat& frame);

private:
    void saveScore();
};

class Game {
private:
    string windowName;
    int WIDTH;
    int HEIGHT;
    bool isRunning;
    int score;
    int lives;
    int round;
    Mat frame;
    GameState* currentState;

public:
    Game() : windowName(WINDOW_NAME), WIDTH(::WIDTH), HEIGHT(::HEIGHT), isRunning(true), score(0), lives(INITIAL_LIVES), round(1), currentState(nullptr) {
        namedWindow(windowName, WINDOW_NORMAL);
        resizeWindow(windowName, WIDTH, HEIGHT);
    }

    ~Game() {
        delete currentState;
        destroyAllWindows();
    }

    void init() {
        cout << "Game::init() chamado!" << endl;
        currentState = new MenuState(this);
        cout << "MenuState criado!" << endl;
        currentState->render(frame); // Chama renderização inicial
    }

    void run() {
        while (isRunning) {
            currentState->handleInput();
            currentState->update();
            currentState->render(frame);
        }
    }

    void changeState(GameState* newState) {
        if (currentState != nullptr) {
            delete currentState;
        }
        currentState = newState;
    }

    int getScore() const {
        return score;
    }

    int getLives() const {
        return lives;
    }

    int getRound() const {
        return round;
    }

    Mat& getFrame() {
        return frame;
    }

    void resetScore() {
        score = 0;
    }

    void addScore(int points) {
        score += points;
    }

    void resetLives() {
        lives = INITIAL_LIVES;
    }

    void decrementLives() {
        lives--;
        if (lives <= 0) {
            changeState(new GameOverState(this));
        }
    }

    void resetRound() {
        round = 1;
    }

    void incrementRound() {
        round++;
    }

    void setRunning(bool running) {
        isRunning = running;
    }
};

void MenuState::handleInput() {
    int key = waitKey(0);
    if (key == 49) { // '1'
        game->changeState(new PlayState(game));
    } else if (key == 50) { // '2'
        showTopScores();
    } else if (key == 27) { // ESC
        game->setRunning(false);
    }
}

void MenuState::update() {}

void MenuState::render(Mat& frame) {
    cout << "MenuState::render() chamado!" << endl;
    Mat menuFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    putText(menuFrame, "BloonsCV6", Point(WIDTH / 4, 100), FONT_HERSHEY_DUPLEX, 2.0, Scalar(0, 255, 255), 3);
    putText(menuFrame, "1 - Jogar", Point(WIDTH / 4, 200), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    putText(menuFrame, "2 - Ver Top 5", Point(WIDTH / 4, 300), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    putText(menuFrame, "ESC - Sair", Point(WIDTH / 4, 400), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 255, 255), 2);
    frame = menuFrame;
    imshow(WINDOW_NAME, frame);
}

void MenuState::showTopScores() {
    vector<pair<string, int>> topScores = scoreManager->getTopScores();
    Mat topFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));

    putText(topFrame, "Top 5 Pontuacoes:", Point(100, 80), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 0), 2);

    for (size_t i = 0; i < topScores.size() && i < 5; ++i) {
        ostringstream oss;
        oss << (i + 1) << ". " << topScores[i].first << " - " << topScores[i].second;
        putText(topFrame, oss.str(), Point(150, 150 + 50 * i),
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
    }

    putText(topFrame, "Pressione qualquer tecla para voltar...",
            Point(100, HEIGHT - 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

    imshow(WINDOW_NAME, topFrame);
    waitKey(0);
}

void PlayState::handleInput() {
    char key = static_cast<char>(waitKey(30));
    if (key == 27 || key == 'q' || key == 'Q') {
        // Perguntar se deseja salvar a pontuação
        Mat prompt = Mat::zeros(200, 640, CV_8UC3);
        putText(prompt, "Deseja salvar sua pontuacao? (S/N)", Point(30, 100), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        imshow("BloonsCV6", prompt);

        // Espera por S ou N
        while (true) {
            char response = (char)waitKey(0);
            if (response == 's' || response == 'S') {
                game->changeState(new GameOverState(game));
                break;
            } else if (response == 'n' || response == 'N') {
                game->changeState(new MenuState(game));
                break;
            }
        }
    }
    if (key == 'i' || key == 'I') {
        faceDetector->toggleCameraFlip();
    }
}

void PlayState::update() {
    faceDetector->detectFaces(game->getFrame(), faces, WIDTH, HEIGHT);

    if (frameCount % SPAWN_INTERVAL == 0) {
        spawnBalloon();
    }

    updateBalloons();
    checkCollisions();
    removeOffScreenBalloons();

    if (game->getScore() >= game->getRound() * ROUND_POINTS_MULTIPLIER) {
        game->incrementRound();
        soundManager->playRoundStartSound();
    }

    frameCount++;
}

void PlayState::render(Mat& frame) {
    drawGameFrame(frame);
    imshow(WINDOW_NAME, frame);
}

void PlayState::spawnBalloon() {
    int side = rand() % 4;
    int maxType = min(game->getRound() - 1, resources->getTotalBalloonTypes() - 1);
    int type = rand() % (maxType + 1);
    float speedMultiplier = 1.0f + (game->getRound() * 0.1f);
    float speed = (5.0f + (type * 5.0f)) * speedMultiplier;
    Point2f pos, vel;

    switch (side) {
        case 0: pos = Point2f(rand() % WIDTH, -BALLOON_RADIUS); vel = Point2f((rand() % 3 - 1) * 0.5f, speed); break;
        case 1: pos = Point2f(rand() % WIDTH, HEIGHT + BALLOON_RADIUS); vel = Point2f((rand() % 3 - 1) * 0.5f, -speed); break;
        case 2: pos = Point2f(-BALLOON_RADIUS, rand() % (HEIGHT / 2) + (HEIGHT / 4)); vel = Point2f(speed, (rand() % 3 - 1) * 0.5f); break;
        case 3: pos = Point2f(WIDTH + BALLOON_RADIUS, rand() % (HEIGHT / 2) + (HEIGHT / 4)); vel = Point2f(-speed, (rand() % 3 - 1) * 0.5f); break;
    }

    balloons.emplace_back(pos, vel, resources->getBalloonSprite(type), resources->getBalloonPoints(type));
    balloons.back().setVisible(false);
}

void PlayState::updateBalloons() {
    for (auto& balloon : balloons) {
        balloon.update();
        if (!balloon.getPoppedState() && !balloon.isVisible() &&
            balloon.position.x >= -BALLOON_RADIUS && balloon.position.x <= WIDTH + BALLOON_RADIUS &&
            balloon.position.y >= -BALLOON_RADIUS && balloon.position.y <= HEIGHT + BALLOON_RADIUS) {
            balloon.setVisible(true);
        }
    }
}

void PlayState::checkCollisions() {
    Mat dardoSprite = resources->getDardo();
    for (auto& balloon : balloons) {
        if (!balloon.getPoppedState() && balloon.isVisible()) {
            for (const auto& face : faces) {
                if (balloon.checkCollision(face)) {
                    balloon.setPoppedState(true);
                    game->addScore(balloon.getPoints());
                    soundManager->playPopSound();
                }
            }
        }
    }
}

void PlayState::removeOffScreenBalloons() {
    balloons.erase(remove_if(balloons.begin(), balloons.end(), [&](Balloon& b) {
        if (!b.getPoppedState()) {
            if (b.isVisible() && b.isOffScreen(WIDTH, HEIGHT)) {
                game->decrementLives();
                return true;
            }
            b.setVisible(b.position.x >= -BALLOON_RADIUS && b.position.x <= WIDTH + BALLOON_RADIUS &&
                          b.position.y >= -BALLOON_RADIUS && b.position.y <= HEIGHT + BALLOON_RADIUS);
        }
        return b.getPoppedState() && b.getFramesSincePop() > POP_ANIMATION_FRAMES;
    }), balloons.end());
}

void PlayState::drawGameFrame(Mat& frame) {
    resources->getBackground().copyTo(frame);
    Mat dardoSprite = resources->getDardo();
    Mat popAnimationSprite = resources->getPopAnimation();

    for (const auto& balloon : balloons) {
        balloon.render(frame, popAnimationSprite);
    }

    for (const auto& face : faces) {
        if (dardoCache.find(face.width) == dardoCache.end()) {
            resize(dardoSprite, dardoCache[face.width], Size(face.width, face.height));
        }
        overlayImage(frame, dardoCache[face.width], frame, Point2f(face.x, face.y));
    }

    putText(frame, "Pontuacao: " + to_string(game->getScore()), Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
    putText(frame, "Vidas: " + to_string(game->getLives()), Point(10, 70),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    putText(frame, "Rodada: " + to_string(game->getRound()), Point(WIDTH - 200, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
}

void GameOverState::handleInput() {
    int key = waitKey(0);
    if (key == 13 || key == 10) { // Enter
        saveScore();
        game->changeState(new MenuState(game));
    } else if (key == 8 && !playerName.empty()) { // Backspace
        playerName.pop_back();
    } else if (key >= 32 && key <= 126) { // Printable characters
        playerName += static_cast<char>(key);
    }
}

void GameOverState::update() {}

void GameOverState::render(Mat& frame) {
    Mat gameOverFrame(HEIGHT, WIDTH, CV_8UC3, Scalar(0, 0, 0));
    putText(gameOverFrame, "GAME OVER!", Point(WIDTH / 4, 150),
            FONT_HERSHEY_DUPLEX, 2.0, Scalar(0, 0, 255), 4);
    putText(gameOverFrame, "Pontuacao Final: " + to_string(game->getScore()),
            Point(WIDTH / 4, 220), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(gameOverFrame, "Digite seu nome:", Point(WIDTH / 4, 300),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
    putText(gameOverFrame, playerName, Point(WIDTH / 4, 350),
            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
    frame = gameOverFrame;
    imshow(WINDOW_NAME, frame);
}

void GameOverState::saveScore() {
    soundManager->playGameOverSound();
    scoreManager->saveScore(playerName, game->getScore());
}

void overlayImage(const Mat& background, const Mat& foreground, Mat& output, Point2f location) {
    background.copyTo(output);

    for (int y = max(static_cast<int>(location.y), 0); y < background.rows; ++y) {
        int fY = y - static_cast<int>(location.y);
        if (fY >= foreground.rows) break;

        for (int x = max(static_cast<int>(location.x), 0); x < background.cols; ++x) {
            int fX = x - static_cast<int>(location.x);
            if (fX >= foreground.cols) break;

            double opacity = static_cast<double>(foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.0;

            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + x * output.channels() + c] =
                    static_cast<unsigned char>(foregroundPx * opacity + backgroundPx * (1.0 - opacity));
            }
        }
    }
}

int main() {
    Game game;
    game.init();
    game.run();
    return 0;
}