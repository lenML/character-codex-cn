import * as raw_data_json from "../data/character_codex.json";
import * as fs from "fs";
import * as path from "path";
import Groq from "groq-sdk";
import OpenAI from "openai";
import * as dotenv from "dotenv";
import { HttpsProxyAgent } from "https-proxy-agent";
import * as lodash from "lodash";
import { tqdm, range } from "@zzkit/tqdm";
import * as throttle from "@jcoreio/async-throttle";

dotenv.config();

const httpAgent = process.env.HTTPS_PROXY
  ? new HttpsProxyAgent(process.env.HTTPS_PROXY)
  : undefined;

// let MODEL_NAME = "";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY, httpAgent });
const openai = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
  httpAgent,
});
// let CURRENT_CLIENT: Groq | OpenAI = groq;

// MODEL_NAME = "llama3-70b-8192";
const deepinfra = new OpenAI({
  apiKey: process.env.DEEP_INFRA_API_KEY,
  baseURL: "https://api.deepinfra.com/v1/openai",
  httpAgent,
});
// MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct";
// MODEL_NAME = "microsoft/WizardLM-2-8x22B";
// MODEL_NAME = "Qwen/Qwen2-72B-Instruct";
// CURRENT_CLIENT = deepinfra;
const deepseek = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com",
});

// 配置多个模型，从这几个端点轮询
const endpoints = [
  {
    client: deepseek,
    params: {
      model: "deepseek-chat",
      top_p: 0.9,
      temperature: 0.5,
      max_tokens: 512,
    },
  },
  // {
  //   client: deepinfra,
  //   params: {
  //     model: "Qwen/Qwen2-72B-Instruct",
  //     top_p: 0.9,
  //     temperature: 0.75,
  //     max_tokens: 512,
  //   },
  // },
  // {
  //   client: deepinfra,
  //   params: {
  //     model: "meta-llama/Meta-Llama-3-70B-Instruct",
  //     top_p: 0.9,
  //     temperature: 0.5,
  //     max_tokens: 512,
  //   },
  // },
  {
    client: openai,
    params: {
      model: "llama3-70b-8192",
      top_p: 0.9,
      temperature: 0.5,
      max_tokens: 512,
    },
  },
  // {
  //   client: deepinfra,
  //   params: {
  //     model: "microsoft/WizardLM-2-8x22B",
  //     top_p: 0.9,
  //     temperature: 0.75,
  //     max_tokens: 512,
  //   },
  // },
];

type Character = {
  media_type: string;
  genre: string;
  character_name: string;
  media_source: string;
  description: string;
  scenario: string;
};

const sys_prompt = fs.readFileSync(
  path.join(__dirname, "../data/sys.prompt.txt"),
  "utf8"
);
const input_prompt = fs.readFileSync(
  path.join(__dirname, "../data/input.prompt.txt"),
  "utf8"
);

const create_input_prompt = (context: Record<keyof any, any>) => {
  let prompt = input_prompt;
  for (const key in context) {
    if (context.hasOwnProperty(key)) {
      prompt = prompt.replace(`{{${key}}}`, context[key]);
    }
  }
  return prompt;
};

// {
//     "media_type": "Webcomics",
//     "genre": "Fantasy Webcomics",
//     "character_name": "Alana",
//     "media_source": "Saga",
//     "description": "Alana is one of the main characters from the webcomic \"Saga.\" She is a strong-willed and fiercely protective mother who is on the run with her family in a war-torn galaxy. The story blends elements of fantasy and science fiction, creating a rich and complex narrative.",
//     "scenario": "You are a fellow traveler in the galaxy needing help, and Alana offers her assistance while sharing stories of her family's struggles and triumphs."
// },
const raw_data = raw_data_json as Character[];

console.log(raw_data.length);

const run_job = async (
  data: Record<keyof any, any>,
  endpoint: (typeof endpoints)[number]
) => {
  const input_prompt = create_input_prompt({
    ...data,
  });
  // console.log(sys_prompt);
  // console.log(input_prompt);
  const { client, params } = endpoint;
  const resp = await client.chat.completions.create({
    messages: [
      {
        role: "system",
        content: sys_prompt,
      },
      {
        role: "user",
        content: input_prompt,
      },
    ],
    temperature: 0.5,
    max_tokens: 512,
    top_p: 1,
    stream: false,
    ...(params as any),
  });
  const output = resp.choices[0].message.content || "";
  // console.log(output);
  fs.writeFileSync("./output.txt", input_prompt + "\n\n" + output);

  return output;
};

const has_chinese = (text: string) => {
  return /[\u4e00-\u9fa5]/.test(text);
};
const has_japanese = (text: string) => {
  return /[\u3040-\u309F]/.test(text) || /[\u30A0-\u30FF]/.test(text);
};
const has_korean = (text: string) => {
  return /[\u3130-\u318f]/.test(text);
};
const has_cyrillic = (text: string) => {
  return /[А-Яа-я]/.test(text);
};
// 阿拉伯文字
const has_arabic = (text: string) => {
  return /[\u0600-\u06FF]/.test(text);
};

const is_pass_answer = (text: string) => {
  if (text.trim() === "") return false;
  if (!has_chinese(text)) return false;
  if (
    has_japanese(text) ||
    has_korean(text) ||
    has_cyrillic(text) ||
    has_arabic(text)
  )
    return false;
  return true;
};

const get_answer_data = (text: string) => {
  // 获取 ```\n...\n``` 中的内容
  const matches = text.matchAll(/```([\s\S]*?)```/gi);
  const match = Array.from(matches).pop();
  if (match) {
    return match[1].trim();
  }
  throw new Error("Failed to get answer data");
};

const run_char_job = async (char: Character, text: string) => {
  if (has_chinese(text)) {
    // 不需要翻译
    // console.log(`Skip translation for ${char.character_name}: ${text}`);
    return text;
  }
  let retry_times = 5;
  while (retry_times > 0) {
    try {
      const index = 5 - retry_times;
      const endpoint = endpoints[index % endpoints.length];
      const resp = await run_job(
        {
          ...char,
          text,
        },
        endpoint
      );
      const answer = get_answer_data(resp);
      if (is_pass_answer(answer)) {
        return answer;
      }
      console.log(`Failed to translate ${char.character_name}: ${text}`);
      console.log(`Retry ${retry_times} times`);
    } catch (error) {
      console.error(error);
      console.log(`Retry ${retry_times} times`);
    }
    retry_times -= 1;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Failed to translate ${char.character_name}: ${text}`);
};

const run_trans_char = async (char: Character) => {
  const input = lodash.cloneDeep(char);
  char.description = await run_char_job(char, char.description);
  char.scenario = await run_char_job(char, char.scenario);
  return {
    char,
    is_changed: !lodash.isEqual(input, char),
  };
};

const main = async () => {
  const output = lodash.cloneDeep(raw_data);

  const save = (throttle as any)(async () => {
    await fs.promises.writeFile(
      path.join(__dirname, "../data/character_codex.json"),
      JSON.stringify(output, null, 2)
    );
  }, 5000);

  for (const i of tqdm(range(output.length))) {
    const char = output[i];

    // 正式翻译逻辑
    const { is_changed } = await run_trans_char(char);
    if (is_changed) {
      await save();
      await new Promise((resolve) => setTimeout(resolve, 300 * Math.random()));
    }
  }
};

main().catch(console.error);
