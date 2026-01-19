import { NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(req: Request) {
  try {
    // 1. Get user input from Frontend (Source, Dest, Weight, Priority, Goal)
    const { orderDetails } = await req.json();

    // 2. Fetch "Source of Truth" data from MongoDB
    const [allCities, networkConfig] = await Promise.all([
      prisma.city.findMany({ orderBy: { id: 'asc' } }),
      prisma.systemConfig.findUnique({ where: { key: 'network_data' } })
    ]);

    if (!networkConfig) {
      return NextResponse.json({ error: "System not initialized. Run seed script." }, { status: 500 });
    }

    // 3. Prepare the "Master Payload" for C++
    const payload = {
      order: orderDetails,
      cities: allCities,
      distMatrix: networkConfig.matrix,
      spokeToHub: networkConfig.spokeToHub,
      goal: orderDetails.goal // "cost" or "time"
    };

    // 4. Resolve the path to your compiled C++ binary
    const binaryPath = path.resolve('./solver/bin/cargo_vista');

    // 5. Wrap the Child Process in a Promise
    const solverResult = await new Promise((resolve, reject) => {
      const child = spawn(binaryPath);

      let output = "";
      let errorData = "";

      // Send JSON payload to C++
      child.stdin.write(JSON.stringify(payload));
      child.stdin.end();

      // Collect data from C++ stdout
      child.stdout.on('data', (data) => {
        output += data.toString();
      });

      // Collect errors from C++ stderr
      child.stderr.on('data', (data) => {
        errorData += data.toString();
      });

      child.on('close', (code) => {
        if (code === 0) {
          try {
            resolve(JSON.parse(output));
          } catch (e) {
            reject(new Error("C++ returned invalid JSON"));
          }
        } else {
          reject(new Error(`C++ Solver failed (Code ${code}): ${errorData}`));
        }
      });
    });

    // 6. Return the optimized result to your React frontend
    return NextResponse.json(solverResult);

  } catch (error: any) {
    console.error("API Error:", error.message);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
