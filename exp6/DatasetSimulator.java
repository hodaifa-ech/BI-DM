public class DatasetSimulator {

    public static void main(String[] args) {
        int numberOfInstancesToGenerate = 15; 
        List<SimulatedInstance> simulatedDataset = new ArrayList<>();
        Random randomGenerator = new Random();
        String[] possibleCategories = {"Alpha", "Beta", "Gamma", "Delta"};

        System.out.println("--- Generating Simulated Dataset ---");

        for (int i = 0; i < numberOfInstancesToGenerate; i++) {
            
            int instanceId = i + 1; 
            double numFeat1 = 10 + (randomGenerator.nextDouble() * 90); 
            double numFeat2 = randomGenerator.nextDouble() * 50;      
            String category = possibleCategories[randomGenerator.nextInt(possibleCategories.length)]; 

            
            SimulatedInstance newInstance = new SimulatedInstance(instanceId, numFeat1, numFeat2, category);

           
            simulatedDataset.add(newInstance);

            System.out.println("Generated and Added: " + newInstance);
        }

        System.out.println("\n--- Simulated Dataset Generation Complete ---");
        System.out.println("Total instances generated: " + simulatedDataset.size());

        
        System.out.println("\n--- Final Dataset Contents ---");
        for (SimulatedInstance instance : simulatedDataset) {
            System.out.println(instance);
        }
        System.out.println("--- End of Dataset ---");
    }
}