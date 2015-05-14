<?php

/**
 * Description of component
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\layout;
use library\layout\elements\element;


abstract class component extends layout{
    protected $elements = array();
    
    public function addElement(element $element){
        $this->elements[] = $element; 
    }
    
    
    
}
