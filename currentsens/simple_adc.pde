
#define SAMPLES_N 1024
uint8_t samples[SAMPLES_N];

#define RAW_SAMPLES_N 100
#define RAW_SAMPLES_STEP 5
uint8_t raw_samples[RAW_SAMPLES_N];

uint8_t ain_io()
{
  uint16_t sensorValue = analogRead(0);
  if (sensorValue < 128)
    return 0;
  if (sensorValue >= 128+512)
    return 255;
  return sensorValue - 128;
}

int ain_sort_cmp(const void *a, const void *b)
{
    const uint8_t *ap = (const uint8_t*)a, *bp = (const uint8_t*)b;
    if (a < b)
      return -1;
    if (a > b)
      return +1;
    return 0;
}

uint32_t ain_ret_median, ain_ret_center, ain_ret;

int ain()
{
  uint8_t range_max = 0, range_min = 0;
  for (int16_t i = 0; i < SAMPLES_N; i++) {
    samples[i] = ain_io();
    if (i == 0 || samples[i] < range_min)
      range_min = samples[i];
    if (i == 0 || samples[i] > range_max)
      range_max = samples[i];
    delayMicroseconds(100);
  }
  for (int16_t i = 0; i < RAW_SAMPLES_N; i++) {
    raw_samples[i] = samples[i*RAW_SAMPLES_STEP];
  }
  int16_t center = (int16_t(range_min)+int16_t(range_max)) / 2;
  qsort(samples, SAMPLES_N, 1, &ain_sort_cmp);
  int16_t median = samples[SAMPLES_N/2];
  ain_ret_median = 0;
  ain_ret_center = 0;
  Serial.print(median);
  Serial.print(" ");
  Serial.print(center);
  Serial.print(" ");
  for (int16_t i = 0; i < SAMPLES_N; i++) {
    ain_ret_median += abs(int16_t(samples[i]) - median);
    ain_ret_center += abs(int16_t(samples[i]) - center);
  }
  ain_ret = ain_ret_median < ain_ret_center ? ain_ret_median : ain_ret_center;
}

void setup()
{
  Serial.begin(38400);
}

void loop()
{
  ain();
  Serial.print(ain_ret >> 6, DEC);
  Serial.print(" ");
  Serial.print(ain_ret_median >> 6, DEC);
  Serial.print(" ");
  Serial.print(ain_ret_center >> 6, DEC);
  for (int i = 0; i < RAW_SAMPLES_N; i++) {
    Serial.print(" ");
    Serial.print(raw_samples[i], DEC);
  }
  Serial.print("\n");
}

