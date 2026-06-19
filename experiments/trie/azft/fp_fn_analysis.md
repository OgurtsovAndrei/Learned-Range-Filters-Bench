# Анализ возникновения False Negative в вероятностных структурах MMPH

Данный документ разбирает сценарий, при котором структура Approx Z-Fast Trie (AZFT), используемая в Space-optimized MMPH (Section 6), может не найти существующий ключ (False Negative) из-за коллизий на промежуточных этапах поиска.

## 1. Теоретический контекст

Согласно статье Monotone Minimal Perfect Hashing, вариант реализации B (Section 6) является вероятностным (Monte Carlo algorithm). Это плата за достижение компактности в $O(n \log \log w)$ бит. Основным источником неопределенности является использование Minimal Perfect Hash (MPH) в сочетании с бинарным поиском по длинам префиксов.

## 2. Сценарий возникновения False Negative (Пример)

Рассмотрим ситуацию, когда мы ищем ключ $x$, который присутствует в наборе данных.

### Исходные данные:

В дереве существует 2 ключа:
- `00001111` (длина 8)
- `1111` (длина 4)

**Искомый паттерн:** `000011` (является префиксом существующего ключа `00001111`)

### Пошаговый процесс поиска (Two-Fattest Binary Search):

**Инициализация:** Диапазон поиска $(a, b] = (0, 6]$

#### Шаг 1 (Ошибка):
1. Алгоритм выбирает fFast - 4
2. Выполняется запрос `mph.Query("0000")`
3. **Проблема:** Префикса "0000" нет в таблице хендлов (он пропущен при сжатии дерева)
4. **Поведение MPH:** Поскольку MPH детерминирован только для ключей из обучающего набора, для "0000" он возвращает случайный индекс $k \in [0, N-1]$
5. **Коллизия:** В ячейке data[k] хранится информация о какой-то другой ноде ("1111"). И по чистому совпадению `hash("0000") == hash("1111")`, алгоритм ошибочно подтверждает наличие узла длины 4

#### Шаги 2, 3:
Дальше еще 2 итерации для диапазонов:
- `(4, 6]` → `2Fast = 6`
- `(4, 5]` → `2Fast = 5`

Оба неуспешные, а следовательно итоговая строка `1111`

#### Результат:
- Бинарный поиск завершается, выдавая ответ в дереве существует префикс `1111`
- **False Negative:** Ключ существует, но AZFT его не нашел

## 3. Почему MMPH тоже выдает ошибку?

Поскольку MMPH (Вариант B) использует AZFT как «навигатор» для определения ранга, ошибка в дереве напрямую транслируется в ошибку хеш-функции:

- **Неверный узел:** Если AZFT выдал ложный узел или не нашел ничего, у MMPH нет данных для вычисления корректного диапазона индексов $[i, j]$
- **Провал валидации:** Даже если в конце выполняется проверка двух ключей в массиве ($S[i]$ и $S[j]$), они не совпадут с искомым клюгом, так как мы пришли в случайную область массива

**Итог:** MMPH возвращает признак «ключ не найден» (или некорректный ранг), что нарушает свойство детерминизма для существующих элементов.

## 4. Обоснование авторов и защита

Авторы статьи признают эту проблему и решают её через управление вероятностью ошибки $\epsilon$:

- **Вероятность коллизии:** Вероятность того, что на любом из $O(\log w)$ шагов бинарного поиска случится коллизия PSig, крайне мала при правильном выборе длины сигнатуры $\gamma$

### Формула надежности:

$$\gamma \ge (\log \log n + \log \log w - \log \epsilon)$$

- **Практический подход:** Если $\gamma = 64$ бит, вероятность того, что вы столкнетесь с описанным выше сценарием, математически ниже, чем вероятность повреждения данных космическими лучами

## 5. Вывод

Описанный пример доказывает, что Approx Z-Fast Trie

- ✗ Она дает False Positives (из-за коллизий на финальном этапе)
- ✗ Она дает False Negatives (из-за коллизий на промежуточных этапах поиска)

## 6. What else? — Memory layout!

At the current moment trie cannot solve the weak prefix search issue.
Now it can only return the node with the biggest extent which has a common prefix.
To solve the weak prefix search issue, trie has to return the correct child of this node.
So we need to store links to children somehow.

## 7. Mitigation in MMPH

Since MMPH is a static structure that must work correctly only for the set of keys it was built on, we can use a **Las Vegas** approach to handle the probabilistic nature of the AZFT:

1. **Validation:** After building an `ApproxZFastTrie` on the bucket delimiters, the MMPH builder performs a full validation pass. It checks that every key from the input set resolves to the correct bucket using the trie.
2. **Deterministic retry:** If validation fails for even a single key (either due to a False Positive or a False Negative in the trie), the entire trie is discarded and rebuilt with a new seed.
3. **Guarantee:** This process repeats until a working trie is found (or a limit like `maxTrieRebuilds = 100` is reached). When construction succeeds, the resulting MMPH is guaranteed to be 100% correct for the original key set.

This strategy converts the Monte Carlo error probability described in the paper into a construction-time overhead, ensuring runtime correctness without needing the explicit correction sets mentioned in Section 5.2 of the MMPH paper. Detailed documentation can be found in `mmph/relative_trie/README.md`.
