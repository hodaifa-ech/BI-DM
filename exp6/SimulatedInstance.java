import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;


class SimulatedInstance {
    private int id;
    private double numericalFeature1;
    private double numericalFeature2;
    private String categoricalFeature;

    // Constructor
    public SimulatedInstance(int id, double numFeat1, double numFeat2, String catFeat) {
        this.id = id;
        this.numericalFeature1 = numFeat1;
        this.numericalFeature2 = numFeat2;
        this.categoricalFeature = catFeat;
    }

    public int getId() {
        return id;
    }

    public double getNumericalFeature1() {
        return numericalFeature1;
    }

    public double getNumericalFeature2() {
        return numericalFeature2;
    }

    public String getCategoricalFeature() {
        return categoricalFeature;
    }

    @Override
    public String toString() {
        return String.format("Instance[ID=%d, Feature1=%.2f, Feature2=%.2f, Category=%s]",
                             id, numericalFeature1, numericalFeature2, categoricalFeature);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SimulatedInstance instance = (SimulatedInstance) o;
        return id == instance.id; // Uniqueness defined by ID
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}