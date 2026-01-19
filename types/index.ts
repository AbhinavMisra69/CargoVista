types/index.ts
export interface City {
  id: number;
  name: string;
  x: number;
  y: number;
}

export interface Order {
  orderId: number;
  sellerId: number;
  source: number;
  destination: number;
  weight: number;
  volume: number;
  priority?: number;
}

/**
 * Represents a Seller who originates orders
 */
export interface Seller {
  sellerId: number;
  location: number; // City ID
  orders: Order[];
}

/**
 * Long-haul carrier moving goods between major Hubs
 */
export interface HubHubCarrier {
  carrierId: number;
  fromHubId: number;
  toHubId: number;
  maxWeight: number;
  maxVolume: number;
  remainingWeight: number;
  remainingVolume: number;
  speed: number;
  pendingWeight: number;
  pendingVolume: number;
  pendingOrders: Order[];
  assignedOrders: Order[];
}

/**
 * Regional carrier moving goods between a Hub and its Spokes
 */
export interface HubSpokeCarrier {
  carrierId: number;
  hubLocationId: number;
  maxWeight: number;
  maxVolume: number;
  speed: number;
  remainingWeight: number;
  remainingVolume: number;
  pendingWeight: number;
  pendingVolume: number;
  assignedOrders: Order[];
  pendingOrders: Order[];
}

/**
 * Point-to-Point Node used in the VRP/Simulated Annealing solver
 */
export interface PPCity {
  id: number;
  demand: number;
  supply: number;
  orderId: number; // defaults to -1 in C++
  isPickup: boolean;
}

/**
 * Vehicle used in the Point-to-Point/Simulated Annealing model
 */
export interface PPCarrier {
  id: number;
  capacity: number;
  load: number;
  depotID: number;
  route: number[]; // Sequence of PPCity indices (or IDs)
}

export interface CarrierRoute {
  hubId: number;
  route: number[]; // Array of City IDs
  totalDistance: number;
  totalWeight: number;
}

/**
 * Result object for route comparisons
 */
export interface OptimizationResult {
  modelName: "Hub and Spoke" | "Point-to-Point" | "Personalized Carrier";
  time: number;
  cost: number;
  route: number[];
}
