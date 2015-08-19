<?php

/**
 * Description of intext
 *
 * @author durso
 */
namespace library\dom\elements\components;
use library\dom\elements\paired;

class intext extends paired{
    protected $text;
    
    public function __construct($tag,$value) {
        parent::__construct();
        $this->tag = $tag;
        if($value){
            $this->setText($value);
        }
    }
    public function changeText($value,$index = true){
        if($index === true){
            assert(!is_null($this->text));
            $index = $this->indexOf($this->text);
        }
        parent::changeText($value,$index);
    }
    public function appendText($value,$index = true){
        if($index === true){
            assert(!is_null($this->text));
            $index = $this->indexOf($this->text);
        }
        parent::appendText($value,$index);
    }
    public function setText($value){
        assert(is_null($this->text));
        $this->text = new text($value);
        $this->addComponent($this->text);
    }
    

}