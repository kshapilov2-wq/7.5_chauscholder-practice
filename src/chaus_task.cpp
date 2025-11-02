#include <iostream>   
#include <iomanip>     
#include <vector>      
#include <string>    
#include <limits>      
#include <random>      
#include <numeric>    
#include <cmath>       
#include <cstdint>     
using namespace std;           

using Matrix = vector<vector<double>>;   
using Vector = vector<double>;           

// Счётчик базовых операций 
struct OpCounter {                // структура для подсчёта операций
	long long mul_add = 0;        // количество «умножение+сложение» 
	long long div_cnt = 0;       // количество делений
	long long hypot_cnt = 0;       // количество вызовов hypot
	long long sqrt_cnt = 0;       // количество вызовов sqrt
	void reset() { mul_add = div_cnt = hypot_cnt = sqrt_cnt = 0; }  // сброс счётчиков
};

// Базовые линейные алгебраические примитивы с возможностью считать операции
static inline double dot(const Vector& a, const Vector& b, OpCounter* ops = nullptr) { // скалярное произведение a·b
	double s = 0.0;                                                       // аккумулируем сумму
	for (size_t i = 0; i < a.size(); ++i) { s += a[i] * b[i]; if (ops) ops->mul_add += 2; } // умножение+сложение считаем как 2 условных операции
	return s;                                                         
}

static inline double norm2(const Vector& a, OpCounter* ops = nullptr) {    // евклидова норма ||a||_2
	double s = 0.0;                                                       // аккумулируем квадрат нормы
	for (double v : a) { s += v * v; if (ops) ops->mul_add += 2; }              // суммируем квадраты
	if (ops) ops->sqrt_cnt++;                                            // считаем один sqrt
	return sqrt(s);                                                    
}

static inline double frob(const Matrix& A, OpCounter* ops = nullptr) {     // норма Фробениуса ||A||_F
	double s = 0.0;                                                       // накапливаем сумму квадратов
	for (const auto& r : A) for (double v : r) { s += v * v; if (ops) ops->mul_add += 2; } // квадраты всех элементов
	if (ops) ops->sqrt_cnt++;                                            // один sqrt
	return sqrt(s);                                                     
}

static inline Matrix eye(size_t n) {                                     // единичная матрица I_n
	Matrix E(n, vector<double>(n, 0.0));                                 // создаём n×n нули
	for (size_t i = 0; i < n; ++i) E[i][i] = 1.0;                                // ставим единицы на диагональ
	return E;                                                           // возвращаем I
}

static inline Matrix transpose(const Matrix& A) {                         // транспонирование матрицы
	size_t n = A.size(), m = A[0].size();                                   // размеры A (n×m)
	Matrix T(m, vector<double>(n, 0.0));                                 // создаём m×n нули
	for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < m; ++j) T[j][i] = A[i][j];    // заполняем транспонировано
	return T;                                                           // возвращаем A^T
}

static inline Matrix mul(const Matrix& A, const Matrix& B, OpCounter* ops = nullptr) { // умножение матриц C=AB
	size_t n = A.size(), m = B.size(), p = B[0].size();                       // размеры (A:n×m, B:m×p)
	Matrix C(n, vector<double>(p, 0.0));                                 // создаём n×p нули
	for (size_t i = 0; i < n; ++i) {                                            // пробегаем строки A
		for (size_t k = 0; k < m; ++k) {                                        // по «среднему» измерению
			double aik = A[i][k];                                         // берём текущий элемент A
			for (size_t j = 0; j < p; ++j) {                                    // пробегаем столбцы B
				C[i][j] += aik * B[k][j]; if (ops) ops->mul_add += 2;          // аккумулируем сумму произведений; считаем операции
			}
		}
	}
	return C;                                                           
}

static inline Vector mul(const Matrix& A, const Vector& x, OpCounter* ops = nullptr) { // умножение матрицы на вектор y=Ax
	size_t n = A.size(), m = A[0].size();                                   // размеры A (n×m)
	Vector y(n, 0.0);                                                    // создаём нулевой вектор n
	for (size_t i = 0; i < n; ++i) {                                            // по строкам
		double s = 0.0;                                                   // локальная сумма
		for (size_t j = 0; j < m; ++j) { s += A[i][j] * x[j]; if (ops) ops->mul_add += 2; } // скалярная строка·x
		y[i] = s;                                                         // сохраняем компоненту
	}
	return y;                                                         
}

// Печать (для малых n)
static inline void printVec(const string& name, const Vector& v) {       // печать вектора
	cout << name << ": [";
	for (size_t i = 0; i < v.size(); ++i) cout << fixed << setprecision(8) << v[i] << (i + 1 < v.size() ? ", " : ""); // выводим элементы
	cout << "]\n";
}

static inline void printMat(const string& name, const Matrix& A) {       // печать матрицы
	cout << name << ":\n";
	for (const auto& r : A) { for (double v : r) cout << setw(14) << fixed << setprecision(8) << v << " "; cout << "\n"; } // прямоугольная сетка
}

// ---------- QR через вращения ----------
struct QRResult {                            // агрегируем всё, что хотим вернуть пользователю
	Matrix Q, R;                             // факторизация A ≈ Q R
	Vector y;                                // вектор y = Q^T b
	Vector x;                                // решение Rx=y (итоговое x)
	OpCounter ops;                           // счётчики операций
	double orthErrFro = 0.0;                 // ||Q^T Q - I||_F — проверка ортогональности Q
	double factErrFro = 0.0;                 // ||A - Q R||_F — проверка факторизации
};

static inline void applyGivensRows(Matrix& A, size_t i, size_t k, double c, double s, size_t jStart, OpCounter* ops) {
	// Применяем вращение к строкам i и k, начиная с столбца jStart (чтобы не портить уже зануленное)
	size_t m = A[0].size();                  // количество столбцов
	for (size_t j = jStart; j < m; ++j) {            // бежим от jStart до конца
		double t1 = c * A[i][j] + s * A[k][j]; if (ops) ops->mul_add += 2;   // новая строка i, линкомб
		double t2 = -s * A[i][j] + c * A[k][j]; if (ops) ops->mul_add += 2;   // новая строка k, линкомб
		A[i][j] = t1; A[k][j] = t2;              // записываем обратно
	}
}

static inline void applyGivensToVec(Vector& b, size_t i, size_t k, double c, double s, OpCounter* ops) {
	// То же вращение к правой части (2-мерной подсистеме компонентов i и k)
	double t1 = c * b[i] + s * b[k]; if (ops) ops->mul_add += 2;             // новая компонента i
	double t2 = -s * b[i] + c * b[k]; if (ops) ops->mul_add += 2;             // новая компонента k
	b[i] = t1; b[k] = t2;                      // сохраняем
}

QRResult qr_givens(const Matrix& Ain, const Vector& b_in, bool buildQ = true) { // QR через вращения; параллельно применяем к b
	QRResult Rz;                           // создаём результат
	Rz.ops.reset();                        // обнуляем счётчики
	size_t n = Ain.size();                 // размерность (квадратная матрица n×n)
	Matrix A = Ain;                        // копируем A — будем превращать в R
	Vector b = b_in;                       // копируем b — будем превращать в y=Q^T b
	Matrix Q = eye(n);                     // начальный Q = I (если нужно накапливать Q)

	for (size_t j = 0; j < n; ++j) {               // для каждого столбца
		for (size_t i = j + 1; i < n; ++i) {         // зануляем элементы ниже диагонали (строки i>j)
			double a = A[j][j];            // текущий диагональный элемент
			double b2 = A[i][j];            // поддиагональный элемент, который хотим занулить
			if (fabs(b2) < 1e-300) continue; // если уже практически ноль — пропускаем
			double r = hypot(a, b2); Rz.ops.hypot_cnt++; // устойчиво считаем r=sqrt(a²+b²), чтобы не переполниться
			if (r == 0.0) continue;          // защита от нулевого r
			double c = a / r; Rz.ops.div_cnt++;  // косинус вращения
			double s = b2 / r; Rz.ops.div_cnt++; // синус вращения

			applyGivensRows(A, j, i, c, s, j, &Rz.ops); // вращаем строки j и i в A, начиная с столбца j
			applyGivensToVec(b, j, i, c, s, &Rz.ops);   // те же коэффициенты применяем к b

			if (buildQ) {                    // если нужно накапливать Q явно
				// Q <- Q * T^T, где T^T — транспонированное вращение (вращаем СТОЛБЦЫ j и i в Q)
				for (size_t col = 0; col < n; ++col) {        // цикл по строкам Q, удобнее так
					double t1 = c * Q[col][j] - s * Q[col][i]; if (&Rz.ops) Rz.ops.mul_add += 2; // формула для столбца j
					double t2 = s * Q[col][j] + c * Q[col][i]; if (&Rz.ops) Rz.ops.mul_add += 2; // формула для столбца i
					Q[col][j] = t1; Q[col][i] = t2;         // записываем
				}
			}
		}
	}

	// Теперь A стала верхнетреугольной R, а b стала y = Q^T b_in
	Vector x(n, 0.0);                         // решение будем писать сюда
	for (int i = int(n) - 1; i >= 0; --i) {            // обратный ход по строкам снизу вверх
		double s = b[i];                     // правую часть будем «очищать» от известных членов
		for (size_t j = i + 1; j < n; ++j) { s -= A[i][j] * x[j]; Rz.ops.mul_add += 2; } // переносим известные слагаемые
		x[i] = s / A[i][i]; Rz.ops.div_cnt++; // делим на диагональный элемент
	}

	Rz.R = A;                                // сохраняем R
	Rz.y = b;                                // сохраняем y = Q^T b
	Rz.x = x;                                // сохраняем решение
	Rz.Q = buildQ ? Q : Matrix();            // сохраняем Q, если строили

	if (buildQ) {                             // диагностические нормы, если Q есть
		Matrix QR = mul(Q, Rz.R, &Rz.ops);   // вычислим Q*R
		Matrix Diff = QR;                    // скопируем для вычитания
		for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) { Diff[i][j] -= Ain[i][j]; Rz.ops.mul_add++; } // Diff = QR - A
		Rz.factErrFro = frob(Diff, &Rz.ops); // ||A-QR||_F

		Matrix QTQ = mul(transpose(Q), Q, &Rz.ops); // Q^T Q
		for (size_t i = 0; i < n; ++i) QTQ[i][i] -= 1.0;     // Q^TQ - I
		Rz.orthErrFro = frob(QTQ, &Rz.ops);         // ||Q^TQ - I||_F
	}
	return Rz;                                     
}

// ---------- QR через Хаусхолдера (отражения) ----------
struct HouseState {                      // для восстановления Q храним векторы отражений
	vector<Vector> v;                    // v_j — единичный, действует на подпространство j..n-1
};

QRResult qr_householder(const Matrix& Ain, const Vector& b_in, bool buildQ = true) { // QR Хаусхолдера
	QRResult Rz;                         // создаём результат
	Rz.ops.reset();                      // сбрасываем счётчики
	size_t n = Ain.size();               // размерность
	Matrix A = Ain;                      // копия A — превратим в R
	Vector b = b_in;                     // копия b — превратим в y=Q^T b
	HouseState HS; HS.v.resize(n);       // подготовим место под векторы отражений

	for (size_t j = 0; j < n; ++j) {            // для каждого столбца
		Vector x(n - j);                    // берём хвост столбца j: строки j..n-1
		for (size_t i = j; i < n; ++i) x[i - j] = A[i][j]; // копируем хвост

		double nx = norm2(x, &Rz.ops);   // норма хвоста
		if (nx < 1e-300) { HS.v[j] = Vector(); continue; } // если хвост нулевой — отражение не нужно

		double sign = (x[0] >= 0.0 ? 1.0 : -1.0); // выбираем знак, чтобы избежать вычитания близких (устойчивость)
		Vector v = x;                    // v = x + sign*||x||*e1
		v[0] += sign * nx; Rz.ops.mul_add++; // прибавили к первой компоненте
		double nv = norm2(v, &Rz.ops);   // норма v
		if (nv < 1e-300) { HS.v[j] = Vector(); continue; } // защита от вырождения
		for (double& t : v) { t /= nv; Rz.ops.div_cnt++; } // нормируем v до единичной длины

		HS.v[j] = v;                        // сохраняем юнит-вектор отражения

		// Применяем P = I - 2 v v^T к A[j:n-1, j:n-1] (только к подматрице)
		for (size_t col = j; col < n; ++col) {  // проходим столбцы правее/включая j
			double s = 0.0;                 // посчитаем s = v^T * (подстолбец A[j:,col])
			for (size_t i = 0; i < v.size(); ++i) { s += v[i] * A[j + i][col]; Rz.ops.mul_add += 2; } // скалярное произведение
			for (size_t i = 0; i < v.size(); ++i) { A[j + i][col] -= 2.0*v[i] * s; Rz.ops.mul_add += 2; } // A := A - 2 v s^T (по столбцу)
		}

		// Ту же операцию применяем к b[j:n-1]
		double sb = 0.0;                    // sb = v^T b_sub
		for (size_t i = 0; i < v.size(); ++i) { sb += v[i] * b[j + i]; Rz.ops.mul_add += 2; } // скалярное произведение
		for (size_t i = 0; i < v.size(); ++i) { b[j + i] -= 2.0*v[i] * sb; Rz.ops.mul_add += 2; } // отражение правой части
	}

	// A стала R (верхнетреугольная), b стала y = Q^T b
	Vector x(n, 0.0);                      // готовим решение
	for (int i = int(n) - 1; i >= 0; --i) {         // обратный ход
		double s = b[i];                     // начинаем с правой части
		for (size_t j = i + 1; j < n; ++j) { s -= A[i][j] * x[j]; Rz.ops.mul_add += 2; } // вычитаем известные
		x[i] = s / A[i][i]; Rz.ops.div_cnt++; // делим на диагональ
	}

	Rz.R = A;                              // сохраняем R
	Rz.y = b;                              // сохраняем y
	Rz.x = x;                              // сохраняем x

	if (buildQ) {                           // если хотим Q явно
		Matrix Q = eye(n);                 // начнём с I
		for (int j = int(n) - 1; j >= 0; --j) {      // применяем отражения в обратном порядке (Q=P1P2... => Q := Pj Q)
			const Vector& v = HS.v[j];     // берём v_j
			if (v.empty()) continue;       // могло не быть отражения
			for (size_t col = 0; col < n; ++col) { // применяем к каждому столбцу Q
				double s = 0.0;              // s = v^T * Q_sub
				for (size_t i = 0; i < v.size(); ++i) { s += v[i] * Q[j + i][col]; Rz.ops.mul_add += 2; } // скалярное
				for (size_t i = 0; i < v.size(); ++i) { Q[j + i][col] -= 2.0*v[i] * s; Rz.ops.mul_add += 2; } // отражение
			}
		}
		Rz.Q = Q;                          // сохраняем Q

		Matrix QR = mul(Q, Rz.R, &Rz.ops); // проверим факторизацию
		Matrix Diff = QR;                  // скопируем
		for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) { Diff[i][j] -= Ain[i][j]; Rz.ops.mul_add++; } // Diff=QR-A
		Rz.factErrFro = frob(Diff, &Rz.ops); // ||A-QR||_F

		Matrix QTQ = mul(transpose(Q), Q, &Rz.ops); // Q^T Q
		for (size_t i = 0; i < n; ++i) QTQ[i][i] -= 1.0;     // Q^TQ - I
		Rz.orthErrFro = frob(QTQ, &Rz.ops);         // ||Q^TQ - I||_F
	}
	return Rz;                                   
}

// ---------- Генерация тестовых матриц/векторов ----------
Matrix hilbert(size_t n) {                        // матрица Гильберта (плохо обусловленная)
	Matrix H(n, vector<double>(n, 0.0));          // создаём n×n нули
	for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) H[i][j] = 1.0 / double(i + j + 1); // H_ij = 1/(i+j+1) (индексация с 0)
	return H;                                   
}

Matrix random_matrix(size_t n, uint64_t seed) {   // случайная матрица (равномерно -1..1)
	mt19937_64 rng(seed);                        // генератор с 64-битным seed
	uniform_real_distribution<double> U(-1.0, 1.0); // равномерное распределение
	Matrix A(n, vector<double>(n, 0.0));          // создаём n×n
	for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) A[i][j] = U(rng); // заполняем
	return A;                                   
}

Vector random_vector(size_t n, uint64_t seed) {   // случайный столбец (равномерно -1..1)
	mt19937_64 rng(seed);                        // генератор
	uniform_real_distribution<double> U(-1.0, 1.0); // равномерное распределение
	Vector x(n, 0.0);                             // создаём n
	for (size_t i = 0; i < n; ++i) x[i] = U(rng);         // заполняем
	return x;                                  
}

// Вспомогательные нормы/оценки для отчёта
double vecNorm2(const Vector& v) { return sqrt(inner_product(v.begin(), v.end(), v.begin(), 0.0)); } // ||v||_2
double backward_error(const Matrix& A, const Vector& x, const Vector& b) { // относительная обратная ошибка
	Vector r = mul(A, x);                         // r = Ax
	for (size_t i = 0; i < r.size(); ++i) r[i] -= b[i];    // r = Ax - b
	double nr = vecNorm2(r);                      // ||r||
	double nA = frob(A);                          // ||A||_F (оценка сверху для ||A||_2)
	double nx = vecNorm2(x);                      // ||x||
	double nb = vecNorm2(b);                      // ||b||
	double denom = nA * nx + nb;                    // знаменатель формулы
	return (denom == 0.0 ? 0.0 : nr / denom);         // избегаем деления на 0
}

// Теоретические флопы для сравнения 
static inline double flops_theory_givens(size_t n) { return 2.0*pow((double)n, 3.0); }   // ≈ 2 n^3
static inline double flops_theory_house(size_t n) { return (4.0 / 3.0)*pow((double)n, 3.0); } // ≈ 4/3 n^3

int main() {                                       
	setlocale(LC_ALL, "Russian");
	cin.tie(nullptr);                             // отвязываем cin от cout

	cout << "QR-разложение (Метод разложений и Хаусхолдер) + решение Ax=b\n";      
	cout << "Режимы:\n1) Ввод A и b вручную\n2) Случайная A, b=A*x*\n3) Гильбертова A, b=A*x*\nВыбор [1/2/3]: "; // меню режимов
	int mode = 2;                                   // по умолчанию — 2 (быстрый тест)
	if (!(cin >> mode)) { cerr << "Некорректный ввод режима\n"; return 0; }   // проверяем ввод

	size_t n;                                     // размерность системы
	cout << "Размерность n: ";                     
	if (!(cin >> n) || n == 0) { cerr << "n должен быть >=1\n"; return 0; }     // валидируем

	Matrix A(n, vector<double>(n, 0.0));           // матрица A
	Vector b(n, 0.0);                              // правая часть b
	Vector x_true;                                 // «истинное» x* (для режимов 2/3)
	uint64_t seedA = 1234567ULL, seedX = 7654321ULL;  // источники случайности

	if (mode == 1) {                                  // режим ручного ввода
		cout << "Введите A (" << n << "x" << n << ") построчно:\n";            
		for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) cin >> A[i][j];  // читаем A
		cout << "Введите b (" << n << "):\n";                                 
		for (size_t i = 0; i < n; ++i) cin >> b[i];                              // читаем b
	}
	else if (mode == 2) {                            // случайная A, b=A*x*
		cout << "Начальные точки для A и x* (по умолч. 1234567 7654321): ";         // просим seed
		if (!(cin >> seedA >> seedX)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); } // безопасный парсинг
		A = random_matrix(n, seedA);               // генерим A
		x_true = random_vector(n, seedX);          // генерим x*
		b = mul(A, x_true);                        // b=A*x*
		cout << "Сгенерированы A и x*, сформировано b=A*x*.\n";          
	}
	else if (mode == 3) {                            // Гильбертова A, b=A*x*
		cout << "начальная точка для x* (по умолч. 7654321): ";                      // просим seed
		if (!(cin >> seedX)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); } // безопасный парсинг
		A = hilbert(n);                            // строим Hilbert(n)
		x_true = random_vector(n, seedX);          // генерим x*
		b = mul(A, x_true);                        // b=A*x*
		cout << "Матрица A — Гильберта; b=A*x*.\n";                        
	}
	else {                                        // неверный режим
		cerr << "Неизвестный режим\n"; return 0;     // выходим
	}

	bool small = (n <= 6);                           // для маленьких n — печатаем матрицы полностью
	cout << fixed << setprecision(8);                 

	if (small) {                                    // если маленькая размерность
		printMat("A", A);                          // печатаем A
		printVec("b", b);                          // печатаем b
		if (!x_true.empty()) printVec("x* (истинный)", x_true); // печатаем x*, если есть
	}

	cout << "\n=== Метод вращения ===\n";        
	QRResult G = qr_givens(A, b, /*buildQ=*/true); // QR вращением, строим Q
	if (small) {                                    // для малых n
		printMat("Q_givens", G.Q);                 // печатаем Q
		printMat("R_givens", G.R);                 // печатаем R
		printVec("y = Q^T b", G.y);                // печатаем y
	}
	printVec("x_givens", G.x);                     // выводим решение

	Vector rG = mul(A, G.x);                       // r = A x
	for (size_t i = 0; i < n; ++i) rG[i] -= b[i];           // r = A x - b
	double resG = vecNorm2(rG);                    // ||r||
	double relResG = resG / (vecNorm2(b) + 1e-300);  // относительная невязка
	double backG = backward_error(A, G.x, b);      // обратная ошибка

	cout << "||Ax-b||_2 (givens) = " << resG << "\n";    // печать нормы невязки
	cout << "rel.residual (givens) = " << relResG << "\n"; // относительная невязка
	cout << "backward error (givens) = " << backG << "\n"; // обратная ошибка
	if (!x_true.empty()) {                            // если есть «истина»
		Vector err = G.x;                           // копия решения
		for (size_t i = 0; i < n; ++i) err[i] -= x_true[i];  // err = x - x*
		double relErr = vecNorm2(err) / (vecNorm2(x_true) + 1e-300); // относит. ошибка
		cout << "rel.error vs x* (givens) = " << relErr << "\n"; 
	}
	if (!G.Q.empty()) {                                // если есть Q
		cout << "||A-QR||_F (givens) = " << G.factErrFro << "\n"; // факторизационная ошибка
		cout << "||Q^TQ - I||_F (givens) = " << G.orthErrFro << "\n"; // неортогональность
	}
	cout << "Флопы (теория) = " << flops_theory_givens(n) << "\n"; // теоретическая оценка
	cout << "Подсчёт (грубо): mul+add=" << G.ops.mul_add << ", div=" << G.ops.div_cnt 
		<< ", hypot=" << G.ops.hypot_cnt << ", sqrt=" << G.ops.sqrt_cnt << "\n";

	cout << "\n=== Хаусхолдер (отражения) ===\n";     
	QRResult H = qr_householder(A, b, /*buildQ=*/true); // QR отражениями, строим Q
	if (small) {                                    // если маленькая размерность
		printMat("Q_house", H.Q);                  // печатаем Q
		printMat("R_house", H.R);                  // печатаем R
		printVec("y = Q^T b", H.y);                // печатаем y
	}
	printVec("x_house", H.x);                      // печатаем решение

	Vector rH = mul(A, H.x);                       // r = A x
	for (size_t i = 0; i < n; ++i) rH[i] -= b[i];           // r = Ax - b
	double resH = vecNorm2(rH);                    // ||r||
	double relResH = resH / (vecNorm2(b) + 1e-300);    // относительная невязка
	double backH = backward_error(A, H.x, b);      // обратная ошибка

	cout << "||Ax-b||_2 (house) = " << resH << "\n";     
	cout << "rel.residual (house) = " << relResH << "\n";
	cout << "backward error (house) = " << backH << "\n";
	if (!x_true.empty()) {                            // если есть «истина»
		Vector err = H.x;                           // копия
		for (size_t i = 0; i < n; ++i) err[i] -= x_true[i];  // err = x - x*
		double relErr = vecNorm2(err) / (vecNorm2(x_true) + 1e-300); // относит. ошибка
		cout << "rel.error vs x* (house) = " << relErr << "\n";        
	}
	if (!H.Q.empty()) {                                // если есть Q
		cout << "||A-QR||_F (house) = " << H.factErrFro << "\n";       // факторизационная ошибка
		cout << "||Q^TQ - I||_F (house) = " << H.orthErrFro << "\n";   // ортогональность
	}
	cout << "Флопы (теория) = " << flops_theory_house(n) << "\n";      // теоретическая оценка
	cout << "Подсчёт (грубо): mul+add=" << H.ops.mul_add << ", div=" << H.ops.div_cnt  // счётчики
		<< ", hypot=" << H.ops.hypot_cnt << ", sqrt=" << H.ops.sqrt_cnt << "\n";

	Vector diff = G.x;                              // сравним решения двух методов
	for (size_t i = 0; i < n; ++i) diff[i] -= H.x[i];        // diff = x_givens - x_house
	cout << "\n||x_givens - x_house||_2 = " << vecNorm2(diff) << "\n"; // норма разности решений

	cout << "||A||_F = " << frob(A) << "\n";              // напечатаем норму Фробениуса A 
	if (small) cout << "\n(Для n<=6 матрицы выведены целиком.)\n"; 
	return 0;                                     
}