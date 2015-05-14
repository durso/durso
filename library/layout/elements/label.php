<?php
/**
 * This class creates a label element
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class label extends element{
    
    private $for;
    
    public function __construct($value, $for = false) {
        $this->value = $value;
        $this->tag = "label";
        $this->closeTag = true;
        $this->attributes["for"] = $for;
    }

}