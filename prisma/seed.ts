import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

const citiesData = [
  { id: 1, name: "Delhi", x: 700, y: 220, isHub: false },
  { id: 2, name: "Amritsar", x: 640, y: 130, isHub: false },
  { id: 3, name: "Chandigarh", x: 670, y: 150, isHub: false },
  { id: 4, name: "Jaipur", x: 600, y: 350, isHub: false },
  { id: 5, name: "Lucknow", x: 690, y: 320, isHub: false },
  { id: 6, name: "Kanpur", x: 675, y: 340, isHub: false },
  { id: 7, name: "Agra", x: 670, y: 275, isHub: false },
  { id: 8, name: "Varanasi", x: 750, y: 375, isHub: false },
  { id: 9, name: "Meerut", x: 720, y: 250, isHub: false },
  { id: 10, name: "Aligarh", x: 690, y: 260, isHub: false },
  { id: 11, name: "Patna", x: 770, y: 410, isHub: false },
  { id: 12, name: "Ghaziabad", x: 715, y: 230, isHub: false },
  { id: 13, name: "Moradabad", x: 730, y: 265, isHub: false },
  { id: 14, name: "Bareilly", x: 705, y: 280, isHub: false },
  { id: 15, name: "Saharanpur", x: 690, y: 205, isHub: false },
  { id: 16, name: "Haridwar", x: 670, y: 190, isHub: false },
  { id: 17, name: "Roorkee", x: 665, y: 180, isHub: false },
  { id: 18, name: "Rishikesh", x: 660, y: 170, isHub: false },
  { id: 19, name: "Nainital", x: 655, y: 150, isHub: false },
  { id: 20, name: "Mathura", x: 680, y: 265, isHub: false },
  { id: 21, name: "Hoshiarpur", x: 645, y: 140, isHub: false },
  { id: 22, name: "Kullu", x: 655, y: 120, isHub: false },
  { id: 23, name: "Shimla", x: 660, y: 110, isHub: false },
  { id: 24, name: "Kangra", x: 650, y: 125, isHub: false },
  { id: 25, name: "Solan", x: 660, y: 130, isHub: false },
  { id: 26, name: "Srinagar", x: 610, y: 90, isHub: false },
  { id: 27, name: "Jammu", x: 625, y: 100, isHub: false },
  { id: 28, name: "Ludhiana", x: 660, y: 180, isHub: false },
  { id: 29, name: "Patiala", x: 650, y: 170, isHub: false },
  { id: 30, name: "Panipat", x: 705, y: 240, isHub: false },
  { id: 31, name: "Sonipat", x: 710, y: 230, isHub: false },
  { id: 32, name: "Muzaffarnagar", x: 720, y: 260, isHub: false },
  { id: 33, name: "Fatehpur", x: 690, y: 310, isHub: false },
  { id: 34, name: "Karnal", x: 705, y: 230, isHub: false },
  { id: 35, name: "Bhiwani", x: 680, y: 245, isHub: false },
  { id: 36, name: "Hisar", x: 670, y: 275, isHub: false },
  { id: 37, name: "Jind", x: 660, y: 280, isHub: false },
  { id: 38, name: "Kurukshetra", x: 715, y: 220, isHub: false },
  { id: 39, name: "Rohtak", x: 720, y: 230, isHub: false },
  { id: 40, name: "Faridabad", x: 700, y: 215, isHub: false },
  { id: 41, name: "Barnala", x: 640, y: 150, isHub: false },
  { id: 42, name: "Muktsar", x: 635, y: 135, isHub: false },
  { id: 43, name: "Sangrur", x: 630, y: 140, isHub: false },
  { id: 44, name: "Bhatinda", x: 625, y: 160, isHub: false },
  { id: 45, name: "Jalandhar", x: 650, y: 160, isHub: false },
  { id: 46, name: "Ambala", x: 675, y: 160, isHub: false },
  { id: 47, name: "Gurugram", x: 710, y: 210, isHub: false },
  { id: 48, name: "Noida", x: 715, y: 220, isHub: false },
  { id: 49, name: "Farukhabad", x: 755, y: 360, isHub: false }
];

async function main() {
  console.log('--- Initializing CargoVista Database ---');
  
  // Wipe existing cities to prevent duplicates on re-seed
  await prisma.city.deleteMany({});
  
  // Batch insert all cities
  await prisma.city.createMany({
    data: citiesData,
  });

  console.log(`Success: ${citiesData.length} cities seeded.`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

import { execSync } from 'child_process';
import { prisma } from '@/lib/prisma';

async function seed() {
  // 1. Run C++ 
  console.log("C++ is calculating the logistics network...");
  const rawData = execSync('./solver/bin/exporter').toString();
  const data = JSON.parse(rawData);

  // 2. Save Matrix and Hub-mapping to MongoDB
  await prisma.systemConfig.upsert({
    where: { key: 'network_data' },
    update: { 
      matrix: data.matrix,
      spokeToHub: data.spokeToHub 
    },
    create: { 
      key: 'network_data', 
      matrix: data.matrix,
      spokeToHub: data.spokeToHub 
    },
  });
  console.log("Database initialized with optimized C++ results.");
}
