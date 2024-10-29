import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
DEFAULT_SYSTEM_PROMPT = "Ты эксперт в сфере задачи NER (Named Entity Recognition)"

# Загрузка модели с использованием FP16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.num_beams = 1
generation_config.do_sample = True
generation_config.temperature = 0.7

print(generation_config)

# inputs = ["Напиши числа от 1 до 10"]

inputs = ['''Выдели в следующем тексте всех персонажей, все организации, все страны, все даты в JSON формате, попробуй думать шаг за шагом:
Черногория к началу Второй мировой войны не имела собственной государственности и входила в состав Королевства Югославия. По итогам раздела Югославии державами «оси» Италия аннексировала основную часть Черногорского Приморья
— Боку Которскую, а на остальной территории, включая Санджак, установила с конца апреля 1941 года оккупационное управление.

Первоначально итальянские власти планировали создать марионеточное черногорское королевство, тесно связанное с Королевством Италия личной унией.
Для этого 12 июля 1941 года собранный при покровительстве Италии так называемый Петровданский сабор принял подготовленную в Риме декларацию о восстановлении независимости Черногории и обращение к итальянскому королю с ходатайством о назначении регента.
 Однако итальянский план был навсегда отложен после начавшегося 13 июля народного восстания в Черногории, возглавленного краевым комитетом КПЮ. В августе 1941 года превосходящие силы итальянских войск 9-й армии подавили восстание и установили контроль над большей частью Черногории.
3 октября 1941 года Бенито Муссолини учредил губернаторство Черногория в качестве военно-оккупационной административной единицы. Подавление восстания не привело к ликвидации движения Сопротивления в Черногории. Оставшиеся повстанческие силы разделились на два враждебных движения:
 народно-освободительное под руководством КПЮ и четническое равногорское, формально возглавляемое Драголюбом (Дражей) Михайловичем. Оба движения преследовали цель воссоздания Югославии, однако КПЮ боролась за освобождение страны и завоевание власти с целью социально-политического
  переустройства по советскому образцу, а четники выступали под антикоммунистическими лозунгами за восстановление монархии.

''']
for query in inputs:
    prompt = tokenizer.apply_chat_template([{
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": query
    }], tokenize=False, add_generation_prompt=True)

    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}

    # Засекаем время перед генерацией
    start_time = time.time()

    output_ids = model.generate(**data, generation_config=generation_config)[0]

    # Засекаем время после генерации
    end_time = time.time()

    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print(query)
    print(output)

    # Вычисляем время ответа
    elapsed_time = end_time - start_time
    print(f"Время ответа: {elapsed_time:.4f} секунд")

    print("==============================")
    print()
