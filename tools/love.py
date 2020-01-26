# print('\n'.join([''.join([('Love_You'[(x - y) % 3] if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (x * 0.05) ** 2 * (
#         y * 0.1) ** 3 <= 0 else ' ') for x in range(-30, 30)]) for y in range(15, -15, -1)]))

# for y in range(15, -15, -1):
#     for x in range(-30, 30):
#         a = (x * 0.05) ** 2 + (y * 0.1) ** 2 - 1
#         print('*' if a ** 3 - (x * 0.05) ** 2 * (y * 0.1) ** 3 <= 0 else ' ', end='')
#     print()

print('\n'.join(''.join(
    '*' if ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (x * 0.05) ** 2 * (y * 0.1) ** 3 <= 0 else ' ' for x in
    range(-30, 30)) for y in range(15, -15, -1)))
