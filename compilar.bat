@echo off
echo Compilando o codigo Assembly...
avr-as -mmcu=atmega328p -o calculadora.o calculadora.asm
if errorlevel 1 goto erro

echo Linkando o objeto...
avr-ld -o calculadora.elf calculadora.o
if errorlevel 1 goto erro

echo Gerando arquivo hex...
avr-objcopy -O ihex calculadora.elf calculadora.hex
if errorlevel 1 goto erro

echo Carregando no Arduino...
avrdude -p atmega328p -c arduino -P COM3 -U flash:w:calculadora.hex
if errorlevel 1 goto erro

echo Compilacao e upload concluidos com sucesso!
goto fim

:erro
echo Ocorreu um erro durante o processo!
pause
exit /b 1

:fim
pause 