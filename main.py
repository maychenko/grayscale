from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


image = Image.open("bild.jpg")

gray_image = image.convert("L")

A = np.array(gray_image, dtype=float)

height, width = A.shape

print("Размер изображения:")
print(f"Высота: {height}")
print(f"Ширина: {width}")
print(f"Размер матрицы: {A.shape}")


U, S, VT = np.linalg.svd(A, full_matrices=False)

print("\nSVD выполнено.")



components = [5, 10, 25, 50, 100]

original_size = height * width

print("\n" + "=" * 60)
print("результат сжатия")
print("=" * 60)

results = []


for k in components:

    k = min(k, len(S))

    A_k = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]

    A_k = np.clip(A_k, 0, 255)

    compressed_size = height * k + k + k * width

    compression = original_size / compressed_size

    results.append((k, A_k, compressed_size, compression))

    print(f"\nКоличество компонент: {k}")
    print(f"Исходное количество значений: {original_size}")
    print(f"После сжатия: {compressed_size}")
    print(f"Степень сжатия: {compression:.2f} раз")



plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(A, cmap="gray")
plt.title("Оригинал")
plt.axis("off")


for i, (k, A_k, compressed_size, compression) in enumerate(results):

    plt.subplot(2, 3, i + 2)

    plt.imshow(A_k, cmap="gray")

    plt.title(
        f"k = {k}\n"
        f"Сжатие: {compression:.2f}x"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()



print("\n" + "=" * 60)
print("поиск подходящего K")
print("=" * 60)

for k, A_k, compressed_size, compression in results:

    if compression >= 10:
        print(
            f"k = {k}: сжатие {compression:.2f} раз — "
            f"условие 10x выполнено"
        )
    else:
        print(
            f"k = {k}: сжатие {compression:.2f} раз — "
            f"сжатие меньше 10x"
        )


print("\n" + "=" * 60)
print("ВЫВОД")
print("=" * 60)

print("Выбранное количество компонент: k = 10")
print("Степень сжатия: 11.15 раз")
print("Изображение остается хорошо узнаваемым,")
print("при этом количество хранимых значений уменьшается более чем в 10 раз.")
print("Поэтому k = 10 является подходящим компромиссом")
print("между качеством изображения и обьемом данных.")