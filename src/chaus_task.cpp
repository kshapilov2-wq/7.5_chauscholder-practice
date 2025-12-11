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

//------------------------------------------------------------
// Счётчик базовых операций
//------------------------------------------------------------
struct OpCounter {
	long long mul_add = 0;   // умножения+сложения (условно считаем по 2 за пару)
	long long div_cnt = 0;   // деления
	long long hypot_cnt = 0;   // вызовы hypot
	long long sqrt_cnt = 0;   // вызовы sqrt
	void reset() { mul_add = div_cnt = hypot_cnt = sqrt_cnt = 0; }
};

//------------------------------------------------------------
// Базовые векторно-матричные операции
//------------------------------------------------------------
static inline double dot(const Vector& a, const Vector& b, OpCounter* ops = nullptr)
{
	double s = 0.0;
	for (size_t i = 0; i < a.size(); ++i) {
		s += a[i] * b[i];
		if (ops) ops->mul_add += 2;
	}
	return s;
}

static inline double norm2(const Vector& a, OpCounter* ops = nullptr)
{
	double s = 0.0;
	for (double v : a) {
		s += v * v;
		if (ops) ops->mul_add += 2;
	}
	if (ops) ops->sqrt_cnt++;
	return sqrt(s);
}

static inline double frob(const Matrix& A, OpCounter* ops = nullptr)
{
	double s = 0.0;
	for (const auto& r : A)
		for (double v : r) {
			s += v * v;
			if (ops) ops->mul_add += 2;
		}
	if (ops) ops->sqrt_cnt++;
	return sqrt(s);
}

static inline Matrix eye(size_t n)
{
	Matrix E(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; ++i) E[i][i] = 1.0;
	return E;
}

static inline Matrix transpose(const Matrix& A)
{
	size_t n = A.size(), m = A[0].size();
	Matrix T(m, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < m; ++j)
			T[j][i] = A[i][j];
	return T;
}

static inline Matrix mul(const Matrix& A, const Matrix& B, OpCounter* ops = nullptr)
{
	size_t n = A.size(), m = B.size(), p = B[0].size();
	Matrix C(n, vector<double>(p, 0.0));
	for (size_t i = 0; i < n; ++i) {
		for (size_t k = 0; k < m; ++k) {
			double aik = A[i][k];
			for (size_t j = 0; j < p; ++j) {
				C[i][j] += aik * B[k][j];
				if (ops) ops->mul_add += 2;
			}
		}
	}
	return C;
}

static inline Vector mul(const Matrix& A, const Vector& x, OpCounter* ops = nullptr)
{
	size_t n = A.size(), m = A[0].size();
	Vector y(n, 0.0);
	for (size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (size_t j = 0; j < m; ++j) {
			s += A[i][j] * x[j];
			if (ops) ops->mul_add += 2;
		}
		y[i] = s;
	}
	return y;
}

//------------------------------------------------------------
// Печать
//------------------------------------------------------------
static inline void printVec(const string& name, const Vector& v)
{
	cout << name << ": [";
	for (size_t i = 0; i < v.size(); ++i)
		cout << fixed << setprecision(8) << v[i] << (i + 1 < v.size() ? ", " : "");
	cout << "]\n";
}

static inline void printMat(const string& name, const Matrix& A)
{
	cout << name << ":\n";
	for (const auto& r : A) {
		for (double v : r)
			cout << setw(14) << fixed << setprecision(8) << v << " ";
		cout << "\n";
	}
}

//------------------------------------------------------------
// Результат QR
//------------------------------------------------------------
struct QRResult {
	Matrix Q, R;
	Vector y;
	Vector x;
	OpCounter ops;
	double orthErrFro = 0.0;   // ||Q^T Q - I||_F
	double factErrFro = 0.0;   // ||A - Q R||_F
};

//------------------------------------------------------------
// Гивенс: применяем G к строкам (k, i) слева: A <- G A
// G = [[c, s],[-s, c]]
//------------------------------------------------------------
static inline void applyGivensRows(Matrix& A, size_t k, size_t i,
	double c, double s, size_t startCol,
	OpCounter* ops)
{
	size_t m = A[0].size();
	for (size_t col = startCol; col < m; ++col) {
		double ak = A[k][col];
		double ai = A[i][col];

		double t1 = c * ak + s * ai;   // новая строка k
		double t2 = -s * ak + c * ai;  // новая строка i
		if (ops) ops->mul_add += 4;

		A[k][col] = t1;
		A[i][col] = t2;
	}
}

// тот же поворот к вектору b
static inline void applyGivensToVec(Vector& b, size_t k, size_t i,
	double c, double s, OpCounter* ops)
{
	double bk = b[k];
	double bi = b[i];

	double t1 = c * bk + s * bi;
	double t2 = -s * bk + c * bi;
	if (ops) ops->mul_add += 4;

	b[k] = t1;
	b[i] = t2;
}

//------------------------------------------------------------
// QR через вращения Гивенса:
// 1) вращениями получаем R и y;
// 2) достраиваем Q = A * R^{-1};
//------------------------------------------------------------
QRResult qr_givens(const Matrix& Ain, const Vector& b_in, bool buildQ = true)
{
	QRResult Rz;
	Rz.ops.reset();
	size_t n = Ain.size();

	Matrix A = Ain;   // сюда будем писать R
	Vector b = b_in;  // сюда будем писать y = Q^T b

	// ---------- 1. прямой ход: делаем R и y ----------
	for (size_t k = 0; k < n; ++k) {
		for (size_t i = k + 1; i < n; ++i) {
			double a = A[k][k];
			double b2 = A[i][k];
			if (fabs(b2) < 1e-300) continue;

			double r = hypot(a, b2); Rz.ops.hypot_cnt++;
			if (r == 0.0) continue;

			double c = a / r;  Rz.ops.div_cnt++;
			double s = b2 / r; Rz.ops.div_cnt++;

			// A <- G A
			applyGivensRows(A, k, i, c, s, k, &Rz.ops);

			// b <- G b
			applyGivensToVec(b, k, i, c, s, &Rz.ops);
		}
	}

	// ---------- 2. обратный ход: решаем Rx = y ----------
	Vector x(n, 0.0);
	for (int i = int(n) - 1; i >= 0; --i) {
		double s = b[i];
		for (size_t j = i + 1; j < n; ++j) {
			s -= A[i][j] * x[j];
			Rz.ops.mul_add += 2;
		}
		x[i] = s / A[i][i];
		Rz.ops.div_cnt++;
	}

	Rz.R = A;   // это наш R
	Rz.y = b;   // это y = Q^T b
	Rz.x = x;   // решение

	// ---------- 3. достраиваем Q через R^{-1}, если нужно ----------
	if (buildQ) {
		// 3.1. строим R^{-1} решением R * X = I по столбцам
		Matrix Rinv(n, vector<double>(n, 0.0));

		for (size_t j = 0; j < n; ++j) {
			// e_j
			Vector e(n, 0.0);
			e[j] = 1.0;

			Vector z(n, 0.0);
			for (int i = int(n) - 1; i >= 0; --i) {
				double s = e[i];
				for (size_t k = i + 1; k < n; ++k) {
					s -= A[i][k] * z[k];
					Rz.ops.mul_add += 2;
				}
				z[i] = s / A[i][i];
				Rz.ops.div_cnt++;
			}

			for (size_t i = 0; i < n; ++i)
				Rinv[i][j] = z[i];
		}

		// 3.2. Q = A_orig * R^{-1}
		Matrix Q = mul(Ain, Rinv, &Rz.ops);
		Rz.Q = Q;

		// ---------- диагностика ----------
		Matrix QR = mul(Q, Rz.R, &Rz.ops);
		Matrix Diff = QR;
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j) {
				Diff[i][j] -= Ain[i][j];
				Rz.ops.mul_add++;
			}
		Rz.factErrFro = frob(Diff, &Rz.ops);

		Matrix QTQ = mul(transpose(Q), Q, &Rz.ops);
		for (size_t i = 0; i < n; ++i) QTQ[i][i] -= 1.0;
		Rz.orthErrFro = frob(QTQ, &Rz.ops);
	}

	return Rz;
}

//------------------------------------------------------------
// QR через отражения Хаусхолдера
//------------------------------------------------------------
struct HouseState {
	vector<Vector> v;
};

QRResult qr_householder(const Matrix& Ain, const Vector& b_in, bool buildQ = true)
{
	QRResult Rz;
	Rz.ops.reset();
	size_t n = Ain.size();

	Matrix A = Ain;
	Vector b = b_in;
	HouseState HS; HS.v.resize(n);

	for (size_t j = 0; j < n; ++j) {
		Vector x(n - j);
		for (size_t i = j; i < n; ++i) x[i - j] = A[i][j];

		double nx = norm2(x, &Rz.ops);
		if (nx < 1e-300) { HS.v[j] = Vector(); continue; }

		double sign = (x[0] >= 0.0 ? 1.0 : -1.0);
		Vector v = x;
		v[0] += sign * nx; Rz.ops.mul_add++;
		double nv = norm2(v, &Rz.ops);
		if (nv < 1e-300) { HS.v[j] = Vector(); continue; }
		for (double& t : v) { t /= nv; Rz.ops.div_cnt++; }

		HS.v[j] = v;

		// A[j:n-1, j:n-1] <- (I - 2 v v^T) A
		for (size_t col = j; col < n; ++col) {
			double s = 0.0;
			for (size_t i = 0; i < v.size(); ++i) {
				s += v[i] * A[j + i][col];
				Rz.ops.mul_add += 2;
			}
			for (size_t i = 0; i < v.size(); ++i) {
				A[j + i][col] -= 2.0 * v[i] * s;
				Rz.ops.mul_add += 2;
			}
		}

		// b[j:n-1] <- (I - 2 v v^T) b
		double sb = 0.0;
		for (size_t i = 0; i < v.size(); ++i) {
			sb += v[i] * b[j + i];
			Rz.ops.mul_add += 2;
		}
		for (size_t i = 0; i < v.size(); ++i) {
			b[j + i] -= 2.0 * v[i] * sb;
			Rz.ops.mul_add += 2;
		}
	}

	// Обратный ход
	Vector x(n, 0.0);
	for (int i = int(n) - 1; i >= 0; --i) {
		double s = b[i];
		for (size_t j = i + 1; j < n; ++j) {
			s -= A[i][j] * x[j];
			Rz.ops.mul_add += 2;
		}
		x[i] = s / A[i][i];
		Rz.ops.div_cnt++;
	}

	Rz.R = A;
	Rz.y = b;
	Rz.x = x;

	if (buildQ) {
		Matrix Q = eye(n);
		for (int j = int(n) - 1; j >= 0; --j) {
			const Vector& v = HS.v[j];
			if (v.empty()) continue;
			for (size_t col = 0; col < n; ++col) {
				double s = 0.0;
				for (size_t i = 0; i < v.size(); ++i) {
					s += v[i] * Q[j + i][col];
					Rz.ops.mul_add += 2;
				}
				for (size_t i = 0; i < v.size(); ++i) {
					Q[j + i][col] -= 2.0 * v[i] * s;
					Rz.ops.mul_add += 2;
				}
			}
		}

		Rz.Q = Q;

		Matrix QR = mul(Q, Rz.R, &Rz.ops);
		Matrix Diff = QR;
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j) {
				Diff[i][j] -= Ain[i][j];
				Rz.ops.mul_add++;
			}
		Rz.factErrFro = frob(Diff, &Rz.ops);

		Matrix QTQ = mul(transpose(Q), Q, &Rz.ops);
		for (size_t i = 0; i < n; ++i) QTQ[i][i] -= 1.0;
		Rz.orthErrFro = frob(QTQ, &Rz.ops);
	}

	return Rz;
}

//------------------------------------------------------------
// Генерация тестовых матриц/векторов
//------------------------------------------------------------
Matrix hilbert(size_t n)
{
	Matrix H(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			H[i][j] = 1.0 / double(i + j + 1);
	return H;
}

Matrix random_matrix(size_t n, uint64_t seed)
{
	mt19937_64 rng(seed);
	uniform_real_distribution<double> U(-1.0, 1.0);
	Matrix A(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; ++i)
		for (size_t j = 0; j < n; ++j)
			A[i][j] = U(rng);
	return A;
}

Vector random_vector(size_t n, uint64_t seed)
{
	mt19937_64 rng(seed);
	uniform_real_distribution<double> U(-1.0, 1.0);
	Vector x(n, 0.0);
	for (size_t i = 0; i < n; ++i) x[i] = U(rng);
	return x;
}

//------------------------------------------------------------
// Нормы/ошибки
//------------------------------------------------------------
double vecNorm2(const Vector& v)
{
	return sqrt(inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

double backward_error(const Matrix& A, const Vector& x, const Vector& b)
{
	Vector r = mul(A, x);
	for (size_t i = 0; i < r.size(); ++i) r[i] -= b[i];
	double nr = vecNorm2(r);
	double nA = frob(A);
	double nx = vecNorm2(x);
	double nb = vecNorm2(b);
	double denom = nA * nx + nb;
	return (denom == 0.0 ? 0.0 : nr / denom);
}

static inline double flops_theory_givens(size_t n)
{
	return 2.0 * pow((double)n, 3.0);
}

static inline double flops_theory_house(size_t n)
{
	return (4.0 / 3.0) * pow((double)n, 3.0);
}

//------------------------------------------------------------
// main
//------------------------------------------------------------
int main()
{
	setlocale(LC_ALL, "Russian");
	cin.tie(nullptr);

	cout << "QR-разложение (Гивенс и Хаусхолдер) + решение Ax=b\n";
	cout << "Режимы:\n1) Ввод A и b вручную\n2) Случайная A, b=A*x*\n3) Гильбертова A, b=A*x*\nВыбор [1/2/3]: ";
	int mode = 2;
	if (!(cin >> mode)) { cerr << "Некорректный ввод режима\n"; return 0; }

	size_t n;
	cout << "Размерность n: ";
	if (!(cin >> n) || n == 0) { cerr << "n должен быть >=1\n"; return 0; }

	Matrix A(n, vector<double>(n, 0.0));
	Vector b(n, 0.0);
	Vector x_true;
	uint64_t seedA = 1234567ULL, seedX = 7654321ULL;

	if (mode == 1) {
		cout << "Введите A (" << n << "x" << n << ") построчно:\n";
		for (size_t i = 0; i < n; ++i)
			for (size_t j = 0; j < n; ++j)
				cin >> A[i][j];
		cout << "Введите b (" << n << "):\n";
		for (size_t i = 0; i < n; ++i) cin >> b[i];
	}
	else if (mode == 2) {
		cout << "Начальные точки для A и x* (по умолч. 1234567 7654321): ";
		if (!(cin >> seedA >> seedX)) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
		}
		A = random_matrix(n, seedA);
		x_true = random_vector(n, seedX);
		b = mul(A, x_true);
		cout << "Сгенерированы A и x*, сформировано b=A*x*.\n";
	}
	else if (mode == 3) {
		cout << "Начальная точка для x* (по умолч. 7654321): ";
		if (!(cin >> seedX)) {
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(), '\n');
		}
		A = hilbert(n);
		x_true = random_vector(n, seedX);
		b = mul(A, x_true);
		cout << "Матрица A — Гильберта; b=A*x*.\n";
	}
	else {
		cerr << "Неизвестный режим\n"; return 0;
	}

	bool small = (n <= 6);
	cout << fixed << setprecision(8);

	if (small) {
		printMat("A", A);
		printVec("b", b);
		if (!x_true.empty()) printVec("x* (истинный)", x_true);
	}

	// ----- Гивенс -----
	cout << "\n=== Метод вращений (Гивенс) ===\n";
	QRResult G = qr_givens(A, b, true);
	if (small) {
		printMat("Q_givens", G.Q);
		printMat("R_givens", G.R);
		printVec("y = Q^T b", G.y);
	}
	printVec("x_givens", G.x);

	Vector rG = mul(A, G.x);
	for (size_t i = 0; i < n; ++i) rG[i] -= b[i];
	double resG = vecNorm2(rG);
	double relResG = resG / (vecNorm2(b) + 1e-300);
	double backG = backward_error(A, G.x, b);

	cout << "||Ax-b||_2 (givens) = " << resG << "\n";
	cout << "rel.residual (givens) = " << relResG << "\n";
	cout << "backward error (givens) = " << backG << "\n";
	if (!x_true.empty()) {
		Vector err = G.x;
		for (size_t i = 0; i < n; ++i) err[i] -= x_true[i];
		double relErr = vecNorm2(err) / (vecNorm2(x_true) + 1e-300);
		cout << "rel.error vs x* (givens) = " << relErr << "\n";
	}
	if (!G.Q.empty()) {
		cout << "||A-QR||_F (givens) = " << G.factErrFro << "\n";
		cout << "||Q^TQ - I||_F (givens) = " << G.orthErrFro << "\n";
	}
	cout << "Флопы (теория) = " << flops_theory_givens(n) << "\n";
	cout << "Подсчёт (грубо): mul+add=" << G.ops.mul_add
		<< ", div=" << G.ops.div_cnt
		<< ", hypot=" << G.ops.hypot_cnt
		<< ", sqrt=" << G.ops.sqrt_cnt << "\n";

	// ----- Хаусхолдер -----
	cout << "\n=== Хаусхолдер (отражения) ===\n";
	QRResult H = qr_householder(A, b, true);
	if (small) {
		printMat("Q_house", H.Q);
		printMat("R_house", H.R);
		printVec("y = Q^T b", H.y);
	}
	printVec("x_house", H.x);

	Vector rH = mul(A, H.x);
	for (size_t i = 0; i < n; ++i) rH[i] -= b[i];
	double resH = vecNorm2(rH);
	double relResH = resH / (vecNorm2(b) + 1e-300);
	double backH = backward_error(A, H.x, b);

	cout << "||Ax-b||_2 (house) = " << resH << "\n";
	cout << "rel.residual (house) = " << relResH << "\n";
	cout << "backward error (house) = " << backH << "\n";
	if (!x_true.empty()) {
		Vector err = H.x;
		for (size_t i = 0; i < n; ++i) err[i] -= x_true[i];
		double relErr = vecNorm2(err) / (vecNorm2(x_true) + 1e-300);
		cout << "rel.error vs x* (house) = " << relErr << "\n";
	}
	if (!H.Q.empty()) {
		cout << "||A-QR||_F (house) = " << H.factErrFro << "\n";
		cout << "||Q^TQ - I||_F (house) = " << H.orthErrFro << "\n";
	}
	cout << "Флопы (теория) = " << flops_theory_house(n) << "\n";
	cout << "Подсчёт (грубо): mul+add=" << H.ops.mul_add
		<< ", div=" << H.ops.div_cnt
		<< ", hypot=" << H.ops.hypot_cnt
		<< ", sqrt=" << H.ops.sqrt_cnt << "\n";

	// Сравнение решений
	Vector diff = G.x;
	for (size_t i = 0; i < n; ++i) diff[i] -= H.x[i];
	cout << "\n||x_givens - x_house||_2 = " << vecNorm2(diff) << "\n";

	cout << "||A||_F = " << frob(A) << "\n";
	if (small) cout << "\n(Для n<=6 матрицы выведены целиком.)\n";

	return 0;
}
